"""IA LIBRAS V5 - letras estáticas e gestos dinâmicos com baixa latência.

Executar: python ialibras_v5.py

Requer o arquivo libras_core.py na mesma pasta. Reutiliza dataset_libras/
e lê modelos da V4 quando não há modelos V5. Treinar novamente é recomendado.
"""
import json
import os
import re
import sys
import threading
import time
import unicodedata
import urllib.request
from collections import Counter
from pathlib import Path

# Não inicializar TensorFlow até ser necessário (gestos dinâmicos).
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import cv2
import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

from libras_core import (
    TOTAL_FEATURES, OLD_SEQUENCE_LENGTH, NEW_SEQUENCE_LENGTH,
    GestureRouter, StaticConsensus, normalize_static, normalize_dynamic,
    legacy_dynamic_sequence, motion_score,
)

ROOT = Path(__file__).resolve().parent
DATASET = Path(os.getenv("LIBRAS_DATA_DIR", str(ROOT / "dataset_libras"))).resolve()
STATIC_DIR = DATASET / "estaticos"
DYNAMIC_DIR = DATASET / "dinamicos"

LANDMARKER_PATH = ROOT / "hand_landmarker.task"
LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)

STATIC_V5 = ROOT / "libras_static_v5.pkl"
STATIC_LABELS_V5 = ROOT / "libras_static_labels_v5.json"
DYNAMIC_V5 = ROOT / "libras_dynamic_v5.keras"
DYNAMIC_LABELS_V5 = ROOT / "libras_dynamic_labels_v5.json"
STATIC_V4 = ROOT / "libras_static.pkl"
STATIC_LABELS_V4 = ROOT / "libras_static_labels.json"
DYNAMIC_V4 = ROOT / "libras_gru.keras"
DYNAMIC_LABELS_V4 = ROOT / "libras_dynamic_labels.json"

CAM_W, CAM_H = 640, 480
PROCESS_W, PROCESS_H = 384, 288
PROCESS_INTERVAL = 0.055  # Até ~18 análises/segundo; se lento, descarta frames antigos.
STATIC_TARGET = 40

CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),(0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),(5,9),(9,13),(13,17)
]


def clean_label(raw):
    value = unicodedata.normalize("NFKD", raw.strip().upper())
    value = "".join(c for c in value if not unicodedata.combining(c))
    value = re.sub(r"[^A-Z0-9]+", "_", value).strip("_")
    return value[:48]


def _save_labels(path, labels):
    path.write_text(json.dumps([str(x) for x in labels], ensure_ascii=False, indent=2), encoding="utf-8")


def _load_labels(path):
    labels = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(labels, list) or not all(isinstance(x, str) for x in labels):
        raise ValueError(f"Lista de classes inválida: {path}")
    return labels


def next_filename(directory, prefix):
    numbers = []
    for file in directory.glob(f"{prefix}_*.npy"):
        match = re.fullmatch(rf"{re.escape(prefix)}_(\d+)\.npy", file.name)
        if match:
            numbers.append(int(match.group(1)))
    return directory / f"{prefix}_{max(numbers, default=0) + 1:03d}.npy"


class FreshCamera:
    """Captura contínua em thread: a IA sempre usa o frame mais recente.

    A imagem antiga é descartada se a inferência demorar; isso reduz lag.
    """
    def __init__(self, index=0):
        if sys.platform == "win32":
            cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
            if not cap.isOpened():
                cap.release()
                cap = cv2.VideoCapture(index)
        else:
            cap = cv2.VideoCapture(index)
        if not cap.isOpened():
            cap.release()
            raise RuntimeError("Câmera indisponível; verifique permissões ou outro programa usando a câmera")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_W)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.cap = cap
        self.lock = threading.Lock()
        self.stop_flag = threading.Event()
        self.frame = None
        self.sequence = 0
        self.failed = False
        self.worker = threading.Thread(target=self._read_loop, daemon=True)
        self.worker.start()

    def _read_loop(self):
        failures = 0
        while not self.stop_flag.is_set():
            ok, frame = self.cap.read()
            if not ok:
                failures += 1
                if failures >= 20:
                    self.failed = True
                    break
                time.sleep(0.02)
                continue
            failures = 0
            with self.lock:
                self.frame = frame
                self.sequence += 1

    def read_latest(self, last_sequence):
        with self.lock:
            if self.frame is None or self.sequence == last_sequence:
                return None, last_sequence
            return self.frame.copy(), self.sequence

    def close(self):
        self.stop_flag.set()
        self.worker.join(timeout=0.8)
        self.cap.release()


def _split_by_recording(y, recordings):
    """Amostras do mesmo vídeo/sessão nunca aparecem em treino e teste.

    Classes com apenas uma gravação ficam só no treino e não são avaliadas.
    """
    rng = np.random.default_rng(42)
    train, test = [], []
    for label in sorted(set(y)):
        sessions = sorted(set(recordings[i] for i, value in enumerate(y) if value == label))
        holdout = rng.choice(sessions) if len(sessions) >= 2 else None
        for i, value in enumerate(y):
            if value != label:
                continue
            (test if holdout is not None and recordings[i] == holdout else train).append(i)
    return np.array(train, dtype=int), np.array(test, dtype=int)


class LibrasAI:
    def __init__(self):
        STATIC_DIR.mkdir(parents=True, exist_ok=True)
        DYNAMIC_DIR.mkdir(parents=True, exist_ok=True)
        self.static_model = None
        self.static_labels = None
        self.static_version = None
        self.dynamic_model = None
        self.dynamic_labels = None
        self.dynamic_version = None
        self.landmarker = None
        self.mp = None
        self.last_timestamp = 0
        print("IA Libras V5 | Inicialização sob demanda")
        print(f"Dados: {DATASET}")

    def initialize_landmarker(self):
        if self.landmarker is not None:
            return
        if not LANDMARKER_PATH.is_file():
            print("Baixando modelo de detecção de mãos (apenas na primeira execução)...")
            tmp = LANDMARKER_PATH.with_name(LANDMARKER_PATH.name + ".download")
            try:
                urllib.request.urlretrieve(LANDMARKER_URL, tmp)
                os.replace(tmp, LANDMARKER_PATH)
            finally:
                if tmp.exists():
                    tmp.unlink()
        try:
            import mediapipe as mp
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision
        except ImportError as exc:
            raise RuntimeError("MediaPipe não instalado. Use: pip install -r requirements_libras.txt") from exc
        self.mp = mp
        opts = vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(LANDMARKER_PATH)),
            running_mode=vision.RunningMode.VIDEO, num_hands=2,
            min_hand_detection_confidence=0.55,
            min_hand_presence_confidence=0.55,
            min_tracking_confidence=0.55,
        )
        self.landmarker = vision.HandLandmarker.create_from_options(opts)
        print("MediaPipe pronto.")

    def process(self, frame):
        small = cv2.resize(frame, (PROCESS_W, PROCESS_H), interpolation=cv2.INTER_AREA)
        rgb = np.ascontiguousarray(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
        image = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb)
        stamp = max(time.monotonic_ns() // 1_000_000, self.last_timestamp + 1)
        self.last_timestamp = stamp
        return self.landmarker.detect_for_video(image, stamp)

    @staticmethod
    def has_hand(results):
        return bool(results and results.hand_landmarks)

    @staticmethod
    def extract_features(results):
        out = np.zeros((2, 21, 3), dtype=np.float32)
        if not results or not results.hand_landmarks:
            return out.reshape(-1)
        used = set()
        for i, hand in enumerate(results.hand_landmarks[:2]):
            side = ""
            try:
                side = results.handedness[i][0].category_name.lower()
            except (IndexError, AttributeError):
                pass
            preference = 0 if side == "left" else 1 if side == "right" else None
            slot = preference if preference is not None and preference not in used else next((j for j in range(2) if j not in used), None)
            if slot is None:
                continue
            used.add(slot)
            out[slot] = np.array([(p.x, p.y, p.z) for p in hand], dtype=np.float32)
        return out.reshape(-1)

    @staticmethod
    def draw_hands(frame, results):
        if not results or not results.hand_landmarks:
            return
        h, w = frame.shape[:2]
        for hand in results.hand_landmarks:
            pts = [(int(p.x * w), int(p.y * h)) for p in hand]
            for a, b in CONNECTIONS:
                cv2.line(frame, pts[a], pts[b], (80, 220, 120), 2, cv2.LINE_AA)
            for point in pts:
                cv2.circle(frame, point, 3, (80, 180, 245), -1, cv2.LINE_AA)

    @staticmethod
    def text_panel(frame, lines, footer="ESPACO = iniciar/parar   Q = sair"):
        h, w = frame.shape[:2]
        height = min(h, 38 + 30 * len(lines) + 28)
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, height), (22, 26, 31), -1)
        cv2.addWeighted(overlay, 0.82, frame, 0.18, 0, frame)
        for n, (message, color) in enumerate(lines):
            cv2.putText(frame, str(message)[:70], (15, 33 + 30*n),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65 if n else 0.8, color, 2, cv2.LINE_AA)
        cv2.putText(frame, footer, (12, h - 12), cv2.FONT_HERSHEY_SIMPLEX,
                    0.48, (240,240,240), 1, cv2.LINE_AA)

    @staticmethod
    def camera_loop(callback, title):
        try:
            cam = FreshCamera()
        except RuntimeError as exc:
            print(f"Erro: {exc}")
            return False
        completed = False
        try:
            previous_number = 0
            frames = 0
            while not cam.failed:
                frame, previous_number = cam.read_latest(previous_number)
                if frame is None:
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        break
                    time.sleep(0.005)
                    continue
                frames += 1
                frame = cv2.flip(frame, 1)
                finished = callback(frame, frames)
                cv2.imshow(title, frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q") or finished is True:
                    completed = finished is True
                    break
                if key == ord(" "):
                    callback(frame, -1)  # atalho ESPAÇO
                elif key in (ord("a"), ord("e"), ord("d"), ord("c"), ord("+"), ord("-")):
                    callback(frame, -key - 2)  # comando de teclado
        finally:
            cam.close()
            cv2.destroyAllWindows()
        return completed

    # ------------------------------ CAPTURA ------------------------------
    def record_static(self):
        self.initialize_landmarker()
        label = clean_label(input("Letra ou número (ex.: A, B, 5): "))
        if not label:
            print("Classe inválida.")
            return
        directory = STATIC_DIR / label
        directory.mkdir(parents=True, exist_ok=True)
        samples = []
        recording = False
        previous = None
        last_save = 0.0
        last_process = 0.0
        last_results = None
        print("ESPAÇO inicia | mantenha a forma da mão e altere um pouco a posição entre amostras | Q cancela")

        def step(frame, index):
            nonlocal recording, previous, last_save, last_results, last_process
            if index == -1:
                recording = True
                previous = None
                return
            if index <= 0:
                return
            now = time.monotonic()
            if now - last_process >= PROCESS_INTERVAL:
                last_process = now
                last_results = self.process(frame)
                if self.has_hand(last_results):
                    raw = self.extract_features(last_results)
                    speed = motion_score(previous, raw) if previous is not None else 0.0
                    if recording and speed < 0.12 and now - last_save >= 0.16:
                        samples.append(raw)
                        last_save = now
                    previous = raw
                else:
                    previous = None
            self.draw_hands(frame, last_results)
            self.text_panel(frame, [
                (f"LETRA / NUMERO: {label}", (255,255,255)),
                (f"{'GRAVANDO' if recording else 'AGUARDANDO ESPACO'}  {len(samples)}/{STATIC_TARGET}", (110,230,130)),
                ("Mantenha o formato; varie distancia e posicao", (230,230,230))
            ], footer="ESPACO = gravar    Q = sair")
            return len(samples) >= STATIC_TARGET

        completed = self.camera_loop(step, "IA Libras V5 - Coletar letra")
        if not completed:
            print("Coleta cancelada. Nada foi salvo.")
            return
        if len(samples) < 15:
            print(f"Amostras insuficientes ({len(samples)}). Nada foi salvo.")
            return
        file = next_filename(directory, "estatico")
        np.save(file, np.asarray(samples, np.float32))
        print(f"Salvo: {file} ({len(samples)} amostras). Grave 3+ sessoes em condicoes diferentes.")

    def record_dynamic(self):
        self.initialize_landmarker()
        label = clean_label(input("Nome do gesto (ex.: OI, BOM_DIA, NEUTRO): "))
        if not label:
            print("Classe inválida.")
            return
        directory = DYNAMIC_DIR / label
        directory.mkdir(parents=True, exist_ok=True)
        samples = []
        recording = False
        finished = False
        last_results = None
        last_process = 0.0
        print("ESPAÇO inicia a gravação; execute o sinal completo; ESPAÇO termina. Q cancela.")

        def step(frame, index):
            nonlocal recording, finished, last_results, samples, last_process
            if index == -1:
                if not recording:
                    recording = True
                    samples = []
                else:
                    recording = False
                    finished = True
                return
            if index <= 0:
                return
            now = time.monotonic()
            if now - last_process >= PROCESS_INTERVAL:
                last_process = now
                last_results = self.process(frame)
                if recording and self.has_hand(last_results):
                    samples.append(self.extract_features(last_results))
            self.draw_hands(frame, last_results)
            self.text_panel(frame, [
                (f"GESTO: {label}", (255,255,255)),
                ("GRAVANDO" if recording else "AGUARDANDO ESPACO", (110,230,130)),
                (f"Amostras: {len(samples)}", (230,230,230))
            ])
            return finished

        completed = self.camera_loop(step, "IA Libras V5 - Coletar gesto")
        if not completed:
            print("Coleta cancelada. Nada foi salvo.")
            return
        if len(samples) < 6:
            print("Gesto muito curto ou cancelado. Nada foi salvo.")
            return
        # Formato .npy idêntico ao da V4 para preservar compatibilidade.
        seq = np.asarray(samples, np.float32)
        if len(seq) > OLD_SEQUENCE_LENGTH:
            positions = np.linspace(0, len(seq)-1, OLD_SEQUENCE_LENGTH).astype(int)
            seq = seq[positions]
        out = np.zeros((OLD_SEQUENCE_LENGTH, TOTAL_FEATURES), np.float32)
        out[:len(seq)] = seq
        file = next_filename(directory, "gesto")
        np.save(file, out)
        print(f"Salvo: {file}. Recomendo 10-20 gravações por gesto e classe NEUTRO.")

    def list_dataset(self):
        print("\nDADOS DISPONÍVEIS")
        for title, folder in (("LETRAS / NÚMEROS", STATIC_DIR), ("GESTOS / PALAVRAS", DYNAMIC_DIR)):
            print(f"\n{title}:")
            for item in sorted(folder.iterdir()):
                if item.is_dir():
                    files = list(item.glob("*.npy"))
                    if files:
                        print(f"  {item.name:<20} {len(files):>3} gravação(ões)")

    def _static_data(self):
        X, y, groups = [], [], []
        for directory in sorted(STATIC_DIR.iterdir()):
            if not directory.is_dir():
                continue
            for file in sorted(directory.glob("*.npy")):
                try:
                    arr = np.load(file, allow_pickle=False)
                    if arr.ndim != 2 or arr.shape[1] != TOTAL_FEATURES:
                        print(f"Ignorando formato inválido: {file}")
                        continue
                    for raw in arr:
                        if not np.any(raw):
                            continue
                        X.append(normalize_static(raw))
                        y.append(directory.name)
                        groups.append(str(file))
                except (OSError, ValueError) as exc:
                    print(f"Erro no dado {file}: {exc}")
        if not X:
            return None
        return np.asarray(X, np.float32), np.array(y), np.array(groups)

    def _dynamic_data(self):
        X, y, recordings = [], [], []
        for directory in sorted(DYNAMIC_DIR.iterdir()):
            if not directory.is_dir():
                continue
            for file in sorted(directory.glob("*.npy")):
                try:
                    arr = np.load(file, allow_pickle=False)
                    if arr.ndim != 2 or arr.shape[1] != TOTAL_FEATURES:
                        print(f"Ignorando formato inválido: {file}")
                        continue
                    if np.count_nonzero(np.any(arr != 0, axis=1)) < 6:
                        continue
                    X.append(normalize_dynamic(arr))
                    y.append(directory.name)
                    recordings.append(str(file))
                except (OSError, ValueError) as exc:
                    print(f"Erro no dado {file}: {exc}")
        if not X:
            return None
        return np.asarray(X, np.float32), np.asarray(y), np.asarray(recordings)

    # ---------------------------- TREINAMENTO ----------------------------
    def train_static(self):
        payload = self._static_data()
        if payload is None:
            print("Nenhum dado de letra/número para treinar.")
            return
        X, y, groups = payload
        labels = sorted(set(y))
        if len(labels) < 2:
            print("São necessárias pelo menos 2 classes estáticas.")
            return
        print(f"\nTreino estático V5: {len(X)} amostras, {len(labels)} classes, {len(set(groups))} sessões.")
        print("Sessões por classe:", dict(Counter({x:len(set(groups[y==x])) for x in labels})))
        encoder = LabelEncoder().fit(labels)
        y_encoded = encoder.transform(y)
        train_idx, test_idx = _split_by_recording(y, groups)
        model = ExtraTreesClassifier(
            n_estimators=140, min_samples_leaf=2, max_features=0.7,
            class_weight="balanced", n_jobs=-1, random_state=42
        )
        model.fit(X[train_idx], y_encoded[train_idx])
        if len(test_idx):
            accuracy = accuracy_score(y_encoded[test_idx], model.predict(X[test_idx]))
            print(f"Acurácia em SESSÕES separadas: {accuracy:.1%} (apenas classes com 2+ sessões)")
        else:
            print("Sem validação independente: grave no mínimo 2 sessões por classe.")
        # Modelo final usa todos os dados, após a avaliação honesta por sessão.
        model.fit(X, y_encoded)
        # Paralelismo ajuda no treino, mas adiciona overhead para cada frame
        # individual da webcam. Uma única thread é mais leve na inferência.
        model.set_params(n_jobs=1)
        joblib.dump(model, STATIC_V5)
        _save_labels(STATIC_LABELS_V5, encoder.classes_)
        self.static_model = model
        self.static_labels = list(encoder.classes_)
        self.static_version = 5
        print(f"Modelo salvo: {STATIC_V5.name}")

    def train_dynamic(self):
        payload = self._dynamic_data()
        if payload is None:
            print("Nenhum dado de gesto/palavra para treinar.")
            return
        X, y, groups = payload
        labels = sorted(set(y))
        if len(labels) < 2:
            print("São necessários pelo menos 2 gestos diferentes.")
            return
        counts = Counter(y)
        print(f"\nTreino dinâmico V5: {len(X)} vídeos; {len(labels)} classes.")
        print("Gravações por gesto:", dict(sorted(counts.items())))
        if min(counts.values()) < 8:
            print("AVISO: 8-15 vídeos variados por classe são recomendados. Poucos vídeos = previsões instáveis.")
        try:
            from tensorflow import keras
        except ImportError as exc:
            print("TensorFlow não disponível. Instale com: pip install -r requirements_libras.txt")
            raise RuntimeError("TensorFlow necessário apenas para treinar/reconhecer gestos") from exc
        encoder = LabelEncoder().fit(labels)
        indexes = encoder.transform(y)
        train_idx, test_idx = _split_by_recording(y, groups)
        # GRU pequena e sequências de 48 frames, sem máscara: muito mais leve que V4.
        model = keras.Sequential([
            keras.layers.Input(shape=(NEW_SEQUENCE_LENGTH, TOTAL_FEATURES)),
            keras.layers.GRU(48, return_sequences=True),
            keras.layers.Dropout(0.22),
            keras.layers.GRU(24),
            keras.layers.Dense(32, activation="relu"),
            keras.layers.Dropout(0.20),
            keras.layers.Dense(len(labels), activation="softmax")
        ])
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
                      loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        can_validate = len(test_idx) and all(label in y[train_idx] for label in labels)
        extra = {}
        monitor = "loss"
        if can_validate:
            extra["validation_data"] = (X[test_idx], indexes[test_idx])
            monitor = "val_loss"
            print("Validação por gravações separadas habilitada.")
        else:
            print("Sem validação independente: grave pelo menos 2 vídeos por gesto.")
        stopper = keras.callbacks.EarlyStopping(monitor=monitor, patience=8,
                                                 restore_best_weights=True, min_delta=0.001)
        model.fit(X[train_idx], indexes[train_idx], epochs=55, batch_size=16,
                  verbose=1, callbacks=[stopper], **extra)
        if can_validate:
            metrics = model.evaluate(X[test_idx], indexes[test_idx], verbose=0)
            print(f"Acurácia por vídeos independentes: {metrics[1]:.1%}")
        model.save(DYNAMIC_V5)
        _save_labels(DYNAMIC_LABELS_V5, encoder.classes_)
        self.dynamic_model = model
        self.dynamic_labels = list(encoder.classes_)
        self.dynamic_version = 5
        print(f"Modelo salvo: {DYNAMIC_V5.name}")

    # ------------------------------ MODELOS ------------------------------
    def load_static(self):
        if self.static_model is not None:
            return True
        for model_file, labels_file, version in (
            (STATIC_V5, STATIC_LABELS_V5, 5), (STATIC_V4, STATIC_LABELS_V4, 4)
        ):
            if model_file.is_file() and labels_file.is_file():
                try:
                    self.static_model = joblib.load(model_file)
                    if hasattr(self.static_model, "set_params"):
                        self.static_model.set_params(n_jobs=1)
                    self.static_labels = _load_labels(labels_file)
                    self.static_version = version
                    print(f"Modelo estático V{version} carregado.")
                    return True
                except (OSError, ValueError, KeyError, TypeError) as exc:
                    print(f"Falha ao abrir {model_file}: {exc}")
        return False

    def load_dynamic(self):
        if self.dynamic_model is not None:
            return True
        for model_file, labels_file, version in (
            (DYNAMIC_V5, DYNAMIC_LABELS_V5, 5), (DYNAMIC_V4, DYNAMIC_LABELS_V4, 4)
        ):
            if not (model_file.is_file() and labels_file.is_file()):
                continue
            try:
                from tensorflow.keras.models import load_model
                self.dynamic_model = load_model(model_file, compile=False)
                self.dynamic_labels = _load_labels(labels_file)
                self.dynamic_version = version
                print(f"Modelo dinâmico V{version} carregado.")
                return True
            except Exception as exc:
                print(f"Falha ao carregar {model_file}: {exc}")
        return False

    def predict_static_proba(self, features):
        if not self.load_static():
            return None
        x = normalize_static(features) if self.static_version == 5 else np.asarray(features, np.float32)
        probabilities = self.static_model.predict_proba(x.reshape(1, -1))[0]
        # Se o estimador só conhecer um subconjunto de classes, mapear corretamente.
        out = np.zeros(len(self.static_labels), dtype=np.float32)
        for i, cls in enumerate(self.static_model.classes_):
            out[int(cls)] = probabilities[i]
        return out

    def predict_dynamic_proba(self, sequence):
        if not self.load_dynamic():
            return None
        x = normalize_dynamic(sequence) if self.dynamic_version == 5 else legacy_dynamic_sequence(sequence)
        # Chamada direta evita o overhead de .predict() por frame/lote unitário.
        output = self.dynamic_model(x[np.newaxis, ...], training=False)
        return np.asarray(output.numpy() if hasattr(output, "numpy") else output, np.float32)[0]

    def realtime(self):
        static_ok = self.load_static()
        dynamic_ok = self.load_dynamic()
        if not (static_ok or dynamic_ok):
            print("Treine algum modelo antes do reconhecimento (opção 4).")
            return
        self.initialize_landmarker()
        router = GestureRouter()
        smoother = StaticConsensus(min_confidence=0.77 if self.static_version == 5 else 0.70)
        mode = "AUTO" if static_ok and dynamic_ok else "ESTATICO" if static_ok else "DINAMICO"
        prediction = "Aguardando mao..."
        confidence = 0.0
        last_type = "NENHUM"
        last_display = 0.0
        last_results = None
        frame_times = []
        frames_analyzed = 0
        no_hand_streak = 0
        last_process = 0.0
        print("\nMODOS: A=automático | E=só letras | D=só movimentos | C=limpar | +/- sensibilidade | Q=sair")
        print("Um gesto é classificado DEPOIS que a mão termina o movimento.")

        def step(frame, index):
            nonlocal mode, prediction, confidence, last_type, last_display
            nonlocal last_results, frames_analyzed, no_hand_streak, last_process
            if index < -1:
                key = -index-2
                if key == ord("a") and static_ok and dynamic_ok:
                    mode = "AUTO"
                elif key == ord("e") and static_ok:
                    mode = "ESTATICO"
                elif key == ord("d") and dynamic_ok:
                    mode = "DINAMICO"
                elif key == ord("c"):
                    router.reset()
                elif key == ord("+"):
                    router.start_threshold = max(0.08, router.start_threshold - 0.03)
                    router.stop_threshold = router.start_threshold / 2
                elif key == ord("-"):
                    router.start_threshold = min(0.36, router.start_threshold + 0.03)
                    router.stop_threshold = router.start_threshold / 2
                prediction, confidence, last_type = "Aguardando...", 0.0, "NENHUM"
                smoother.reset()
                router.reset()
                return
            if index <= 0:
                return
            now = time.monotonic()
            frame_times.append(now)
            if len(frame_times) > 45:
                del frame_times[0]
            if now - last_process >= PROCESS_INTERVAL:
                last_process = now
                frames_analyzed += 1
                last_results = self.process(frame)
                if self.has_hand(last_results):
                    no_hand_streak = 0
                    raw = self.extract_features(last_results)
                    event = router.feed(raw, now, mode)
                    if router.moving or router.score > router.stop_threshold * 1.1:
                        smoother.reset()
                        if now - last_display > 0.3:
                            prediction, confidence, last_type = "Analisando movimento...", 0, "MOVIMENTO"
                    if event and event.kind == "static" and static_ok:
                        try:
                            probs = self.predict_static_proba(event.features)
                            result = smoother.update(self.static_labels, probs)
                            if result:
                                label, score = result
                                if label not in ("NEUTRO", "NENHUM", "SEM_SINAL"):
                                    prediction, confidence, last_type = label, 100 * score, "LETRA/NUMERO"
                                    last_display = now
                                else:
                                    prediction, confidence = "Sem sinal definido", 0
                        except Exception as exc:
                            print(f"Falha no modelo estático: {exc}")
                    if event and event.kind == "dynamic" and dynamic_ok:
                        smoother.reset()
                        try:
                            probs = self.predict_dynamic_proba(event.features)
                            winner = int(np.argmax(probs))
                            top = float(probs[winner])
                            margin = top - float(np.partition(probs, -2)[-2]) if len(probs) > 1 else top
                            if (top >= 0.86 and margin >= 0.20
                                    and self.dynamic_labels[winner] not in ("NEUTRO", "NENHUM", "SEM_SINAL")):
                                prediction = self.dynamic_labels[winner]
                                confidence = 100 * top
                                last_type = "GESTO/PALAVRA"
                                last_display = now
                            else:
                                prediction, confidence, last_type = "Gesto nao confirmado", 0, "MOVIMENTO"
                                last_display = now
                        except Exception as exc:
                            print(f"Falha no modelo dinâmico: {exc}")
                else:
                    no_hand_streak += 1
                    if no_hand_streak >= 2:
                        router.feed(None, now, mode)
                        smoother.reset()
                        prediction, confidence, last_type = "Aguardando mao...", 0, "NENHUM"
            # Após um resultado, não manter indefinidamente a letra anterior.
            if last_type in ("LETRA/NUMERO", "GESTO/PALAVRA") and now - last_display > 2.5:
                prediction, confidence, last_type = "Aguardando sinal...", 0, "NENHUM"
            self.draw_hands(frame, last_results)
            display_fps = ((len(frame_times)-1) / (frame_times[-1]-frame_times[0])) if len(frame_times) >= 2 and frame_times[-1] > frame_times[0] else 0.0
            self.text_panel(frame, [
                (f"SINAL: {prediction}", (110,230,130)),
                (f"TIPO: {last_type}  |  MODO: {mode}", (245,245,245)),
                (f"CONFIANCA: {confidence:.0f}%  |  FPS: {display_fps:.1f}", (245,245,245)),
                (f"MOVIMENTO: {router.score:.2f}  |  LIMIAR: {router.start_threshold:.2f}", (220,220,220)),
            ], footer="A=Auto  E=Letras  D=Movimento  +/-=Sensib.  C=Limpar  Q=Sair")

        self.camera_loop(step, "IA Libras V5 - Reconhecimento")

    def delete_class(self):
        option = input("Excluir (1) letra/número ou (2) gesto? ").strip()
        root = STATIC_DIR if option == "1" else DYNAMIC_DIR if option == "2" else None
        if root is None:
            print("Opção inválida")
            return
        label = clean_label(input("Nome da classe para excluir: "))
        target = root / label
        if not target.is_dir() or not label:
            print("Classe não encontrada.")
            return
        if input(f"Excluir todos os arquivos de {label}? Escreva EXCLUIR: ").strip().upper() != "EXCLUIR":
            print("Cancelado.")
            return
        import shutil
        shutil.rmtree(target)
        print("Classe excluída. Treine novamente para atualizar o modelo.")

    def close(self):
        if self.landmarker is not None:
            self.landmarker.close()
            self.landmarker = None


def main():
    ai = LibrasAI()
    try:
        while True:
            print("\n" + "="*56 + "\nIA LIBRAS V5\n" + "="*56)
            print("1. Gravar letra/número estático")
            print("2. Gravar gesto/palavra dinâmico")
            print("3. Ver dados cadastrados")
            print("4. Treinar IA (letras e gestos)")
            print("5. Reconhecimento em tempo real")
            print("6. Excluir classe")
            print("7. Sair")
            option = input("Escolha: ").strip()
            try:
                if option == "1": ai.record_static()
                elif option == "2": ai.record_dynamic()
                elif option == "3": ai.list_dataset()
                elif option == "4":
                    ai.train_static()
                    ai.train_dynamic()
                elif option == "5": ai.realtime()
                elif option == "6": ai.delete_class()
                elif option == "7": break
                else: print("Opção inválida.")
            except (RuntimeError, OSError, ValueError) as exc:
                print(f"Não foi possível concluir: {exc}")
    except KeyboardInterrupt:
        print("\nInterrompido.")
    finally:
        ai.close()


if __name__ == "__main__":
    main()