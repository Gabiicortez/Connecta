# ============================================================
# IA LIBRAS - V4
# RECONHECIMENTO DE LETRAS, NÚMEROS, GESTOS E PALAVRAS
# ============================================================

import cv2
import mediapipe as mp
import numpy as np
import os
import json
import time
import urllib.request
import shutil
import joblib

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import GRU, Dense, Dropout, Masking
from tensorflow.keras.utils import to_categorical

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# ============================================================
# CONFIGURAÇÕES
# ============================================================

MODEL_PATH = "hand_landmarker.task"

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/"
    "hand_landmarker.task"
)

DATASET_DIR = "dataset_libras"

STATIC_DATASET_DIR = os.path.join(
    DATASET_DIR,
    "estaticos"
)

DYNAMIC_DATASET_DIR = os.path.join(
    DATASET_DIR,
    "dinamicos"
)

STATIC_MODEL_FILE = "libras_static.pkl"
DYNAMIC_MODEL_FILE = "libras_gru.keras"

STATIC_LABELS_FILE = "libras_static_labels.json"
DYNAMIC_LABELS_FILE = "libras_dynamic_labels.json"

CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 30

PROCESS_EVERY = 3

PROCESS_WIDTH = 320
PROCESS_HEIGHT = 240

PREDICT_EVERY = 5

SEQUENCE_LENGTH = 60

NUM_HANDS = 2

LANDMARKS_PER_HAND = 21
VALUES_PER_LANDMARK = 3

FEATURES_PER_HAND = (
    LANDMARKS_PER_HAND *
    VALUES_PER_LANDMARK
)

TOTAL_FEATURES = (
    NUM_HANDS *
    FEATURES_PER_HAND
)

MIN_CONFIDENCE = 0.60

STATIC_SAMPLES = 40

MOTION_THRESHOLD = 0.018


# ============================================================
# CLASSE PRINCIPAL
# ============================================================

class LibrasAI:

    def __init__(self):

        print()
        print("=" * 65)
        print("                    IA LIBRAS V4")
        print("=" * 65)
        print()
        print("Inicializando sistema...")

        self.download_model()

        # ----------------------------------------------------
        # MediaPipe
        # ----------------------------------------------------

        base_options = python.BaseOptions(
            model_asset_path=MODEL_PATH
        )

        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_hands=NUM_HANDS,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )

        self.hand_landmarker = (
            vision.HandLandmarker.create_from_options(
                options
            )
        )

        # ----------------------------------------------------
        # Modelo estático
        # ----------------------------------------------------

        self.static_model = None
        self.static_label_encoder = None

        # ----------------------------------------------------
        # Modelo dinâmico
        # ----------------------------------------------------

        self.dynamic_model = None
        self.dynamic_label_encoder = None

        # ----------------------------------------------------
        # Controle
        # ----------------------------------------------------

        self.timestamp = 0
        self.last_results = None

        # ----------------------------------------------------
        # Pastas
        # ----------------------------------------------------

        os.makedirs(
            STATIC_DATASET_DIR,
            exist_ok=True
        )

        os.makedirs(
            DYNAMIC_DATASET_DIR,
            exist_ok=True
        )

        print()
        print("✓ MediaPipe inicializado.")
        print(
            f"✓ Câmera: "
            f"{CAMERA_WIDTH}x{CAMERA_HEIGHT}"
        )
        print(
            f"✓ Processamento: "
            f"{PROCESS_WIDTH}x{PROCESS_HEIGHT}"
        )
        print(
            f"✓ MediaPipe a cada "
            f"{PROCESS_EVERY} frames"
        )
        print(
            f"✓ Predição a cada "
            f"{PREDICT_EVERY} análises"
        )
        print(
            f"✓ Sequência: "
            f"{SEQUENCE_LENGTH} frames"
        )
        print(
            f"✓ Features por frame: "
            f"{TOTAL_FEATURES}"
        )

    # ========================================================
    # DOWNLOAD DO MODELO
    # ========================================================

    def download_model(self):

        if os.path.exists(MODEL_PATH):
            return

        print()
        print(
            "Modelo MediaPipe não encontrado."
        )

        print(
            "Baixando hand_landmarker.task..."
        )

        try:

            urllib.request.urlretrieve(
                MODEL_URL,
                MODEL_PATH
            )

            print(
                "✓ Modelo MediaPipe baixado."
            )

        except Exception as error:

            print()
            print(
                "✗ Erro ao baixar o modelo."
            )

            print(error)

            raise

    # ========================================================
    # ABRIR CÂMERA
    # ========================================================

    def open_camera(self):

        camera = cv2.VideoCapture(
            0,
            cv2.CAP_DSHOW
        )

        if not camera.isOpened():

            camera.release()

            camera = cv2.VideoCapture(
                0
            )

        if not camera.isOpened():

            return None

        camera.set(
            cv2.CAP_PROP_FRAME_WIDTH,
            CAMERA_WIDTH
        )

        camera.set(
            cv2.CAP_PROP_FRAME_HEIGHT,
            CAMERA_HEIGHT
        )

        camera.set(
            cv2.CAP_PROP_FPS,
            CAMERA_FPS
        )

        camera.set(
            cv2.CAP_PROP_BUFFERSIZE,
            1
        )

        camera.set(
            cv2.CAP_PROP_FOURCC,
            cv2.VideoWriter_fourcc(
                *"MJPG"
            )
        )

        return camera

    # ========================================================
    # TIMESTAMP
    # ========================================================

    def reset_timestamp(self):

        self.timestamp = 0

    def get_timestamp(self):

        timestamp = (
            time.monotonic_ns()
            // 1_000_000
        )

        if timestamp <= self.timestamp:

            timestamp = (
                self.timestamp + 1
            )

        self.timestamp = timestamp

        return timestamp

    # ========================================================
    # PROCESSAR FRAME
    # ========================================================

    def process_frame(self, frame):

        small_frame = cv2.resize(
            frame,
            (
                PROCESS_WIDTH,
                PROCESS_HEIGHT
            ),
            interpolation=cv2.INTER_AREA
        )

        rgb = cv2.cvtColor(
            small_frame,
            cv2.COLOR_BGR2RGB
        )

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )

        timestamp = self.get_timestamp()

        results = (
            self.hand_landmarker
            .detect_for_video(
                mp_image,
                timestamp
            )
        )

        self.last_results = results

        return results

    # ========================================================
    # EXTRAIR UMA MÃO
    # ========================================================

    def extract_hand(self, landmarks):

        values = []

        for landmark in landmarks:

            values.append(
                landmark.x
            )

            values.append(
                landmark.y
            )

            values.append(
                landmark.z
            )

        return np.array(
            values,
            dtype=np.float32
        )

    # ========================================================
    # EXTRAIR FEATURES DAS DUAS MÃOS
    # ========================================================

    def extract_features(self, results):

        left_hand = np.zeros(
            FEATURES_PER_HAND,
            dtype=np.float32
        )

        right_hand = np.zeros(
            FEATURES_PER_HAND,
            dtype=np.float32
        )

        if results is None:

            return np.concatenate(
                [
                    left_hand,
                    right_hand
                ]
            )

        if not results.hand_landmarks:

            return np.concatenate(
                [
                    left_hand,
                    right_hand
                ]
            )

        for index, hand_landmarks in enumerate(
            results.hand_landmarks
        ):

            features = self.extract_hand(
                hand_landmarks
            )

            try:

                handedness = (
                    results.handedness[index][0]
                )

                category = (
                    handedness.category_name.lower()
                )

            except Exception:

                category = ""

            if category == "left":

                left_hand = features

            elif category == "right":

                right_hand = features

            else:

                if np.count_nonzero(
                    left_hand
                ) == 0:

                    left_hand = features

                else:

                    right_hand = features

        return np.concatenate(
            [
                left_hand,
                right_hand
            ]
        )

    # ========================================================
    # VERIFICAR SE EXISTE MÃO
    # ========================================================

    def has_hand(self, results):

        if results is None:

            return False

        if not results.hand_landmarks:

            return False

        return True

    # ========================================================
    # DESENHAR MÃOS
    # ========================================================

    def draw_hands(
        self,
        frame,
        results
    ):

        if results is None:
            return

        if not results.hand_landmarks:
            return

        height, width, _ = frame.shape

        connections = [
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 4),

            (0, 5),
            (5, 6),
            (6, 7),
            (7, 8),

            (0, 9),
            (9, 10),
            (10, 11),
            (11, 12),

            (0, 13),
            (13, 14),
            (14, 15),
            (15, 16),

            (0, 17),
            (17, 18),
            (18, 19),
            (19, 20),

            (5, 9),
            (9, 13),
            (13, 17)
        ]

        for hand in results.hand_landmarks:

            points = []

            for landmark in hand:

                x = int(
                    landmark.x * width
                )

                y = int(
                    landmark.y * height
                )

                points.append(
                    (x, y)
                )

            for start, end in connections:

                cv2.line(
                    frame,
                    points[start],
                    points[end],
                    (0, 255, 0),
                    2
                )

            for point in points:

                cv2.circle(
                    frame,
                    point,
                    4,
                    (0, 0, 255),
                    -1
                )

    # ========================================================
    # NORMALIZAR SEQUÊNCIA
    # ========================================================

    def normalize_sequence(
        self,
        sequence
    ):

        sequence = np.array(
            sequence,
            dtype=np.float32
        )

        if len(sequence) < SEQUENCE_LENGTH:

            padding = np.zeros(
                (
                    SEQUENCE_LENGTH
                    - len(sequence),
                    TOTAL_FEATURES
                ),
                dtype=np.float32
            )

            sequence = np.vstack(
                [
                    sequence,
                    padding
                ]
            )

        elif len(sequence) > SEQUENCE_LENGTH:

            indexes = np.linspace(
                0,
                len(sequence) - 1,
                SEQUENCE_LENGTH
            ).astype(int)

            sequence = sequence[
                indexes
            ]

        return sequence

    # ========================================================
    # PRÓXIMO NÚMERO DE ARQUIVO
    # ========================================================

    def next_file_number(
        self,
        folder
    ):

        if not os.path.exists(folder):

            return 1

        files = [
            f
            for f in os.listdir(folder)
            if f.endswith(".npy")
        ]

        return len(files) + 1

    # ========================================================
    # GRAVAR LETRA OU NÚMERO ESTÁTICO
    # ========================================================

    def record_static(self):

        print()
        print("=" * 65)
        print(
            "              GRAVAR LETRA / NÚMERO"
        )
        print("=" * 65)

        print()
        print(
            "Exemplos: A, B, C, D, 1, 2, 3..."
        )

        print()

        label = input(
            "Digite a letra ou número: "
        ).strip().upper()

        if not label:

            print(
                "✗ Nome inválido."
            )

            return

        label_dir = os.path.join(
            STATIC_DATASET_DIR,
            label
        )

        os.makedirs(
            label_dir,
            exist_ok=True
        )

        print()
        print(
            "COMO GRAVAR"
        )

        print("-" * 65)

        print(
            "1. Pressione ESPAÇO."
        )

        print(
            "2. Mantenha a mão parada."
        )

        print(
            "3. O sistema coletará os exemplos."
        )

        print(
            "4. Pressione Q para cancelar."
        )

        print()

        input(
            "Pressione ENTER para abrir a câmera..."
        )

        cap = self.open_camera()

        if cap is None:

            print(
                "✗ Não foi possível abrir a câmera."
            )

            return

        samples = []

        camera_frame = 0

        self.reset_timestamp()

        self.last_results = None

        recording = False

        while True:

            ret, frame = cap.read()

            if not ret:
                break

            camera_frame += 1

            frame = cv2.flip(
                frame,
                1
            )

            if (
                camera_frame
                % PROCESS_EVERY
                == 0
            ):

                results = self.process_frame(
                    frame
                )

                if (
                    recording
                    and
                    self.has_hand(results)
                ):

                    features = (
                        self.extract_features(
                            results
                        )
                    )

                    samples.append(
                        features
                    )

            else:

                results = self.last_results

            self.draw_hands(
                frame,
                results
            )

            status = (
                "COLETANDO"
                if recording
                else "AGUARDANDO"
            )

            cv2.rectangle(
                frame,
                (0, 0),
                (CAMERA_WIDTH, 145),
                (0, 0, 0),
                -1
            )

            cv2.putText(
                frame,
                f"Classe: {label}",
                (15, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                status,
                (15, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                (
                    f"Exemplos: "
                    f"{len(samples)}/"
                    f"{STATIC_SAMPLES}"
                ),
                (15, 105),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                (
                    "ESPACO = iniciar | "
                    "Q = cancelar"
                ),
                (15, 135),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (220, 220, 220),
                1
            )

            cv2.imshow(
                "IA Libras V4 - Estatico",
                frame
            )

            key = cv2.waitKey(1) & 0xFF

            if key == ord(" "):

                recording = True

                print(
                    "● Coleta iniciada."
                )

            elif key == ord("q"):

                samples = []

                break

            if len(samples) >= STATIC_SAMPLES:

                break

        cap.release()

        cv2.destroyAllWindows()

        if len(samples) < 10:

            print(
                "✗ Poucos exemplos coletados."
            )

            return

        number = self.next_file_number(
            label_dir
        )

        filename = os.path.join(
            label_dir,
            f"estatico_{number:03d}.npy"
        )

        np.save(
            filename,
            np.array(
                samples,
                dtype=np.float32
            )
        )

        print()
        print(
            "✓ Exemplos estáticos salvos."
        )

        print(
            f"✓ Classe: {label}"
        )

        print(
            f"✓ Exemplos: {len(samples)}"
        )

        print(
            f"✓ Arquivo: {filename}"
        )

    # ========================================================
    # GRAVAR GESTO DINÂMICO
    # ========================================================

    def record_dynamic(self):

        print()
        print("=" * 65)
        print(
            "                 GRAVAR GESTO"
        )
        print("=" * 65)

        print()
        print("Exemplos:")
        print("OI")
        print("OBRIGADO")
        print("BOM_DIA")
        print("TUDO_BEM")
        print("SIM")
        print("NAO")

        print()

        label = input(
            "Digite o nome do gesto: "
        ).strip().upper()

        if not label:

            print(
                "✗ Nome inválido."
            )

            return

        label_dir = os.path.join(
            DYNAMIC_DATASET_DIR,
            label
        )

        os.makedirs(
            label_dir,
            exist_ok=True
        )

        print()
        print(
            "COMO GRAVAR"
        )

        print("-" * 65)

        print(
            "1. Pressione ESPAÇO para começar."
        )

        print(
            "2. Faça o gesto completo."
        )

        print(
            "3. Pressione ESPAÇO para terminar."
        )

        print(
            "4. Pressione Q para cancelar."
        )

        print()

        input(
            "Pressione ENTER para abrir a câmera..."
        )

        cap = self.open_camera()

        if cap is None:

            print(
                "✗ Não foi possível abrir a câmera."
            )

            return

        sequence = []

        recording = False

        camera_frame = 0

        self.reset_timestamp()

        self.last_results = None

        while True:

            ret, frame = cap.read()

            if not ret:
                break

            camera_frame += 1

            frame = cv2.flip(
                frame,
                1
            )

            if (
                camera_frame
                % PROCESS_EVERY
                == 0
            ):

                results = self.process_frame(
                    frame
                )

                if (
                    recording
                    and
                    self.has_hand(results)
                ):

                    features = (
                        self.extract_features(
                            results
                        )
                    )

                    sequence.append(
                        features
                    )

            else:

                results = self.last_results

            self.draw_hands(
                frame,
                results
            )

            status = (
                "GRAVANDO"
                if recording
                else "AGUARDANDO"
            )

            cv2.rectangle(
                frame,
                (0, 0),
                (CAMERA_WIDTH, 155),
                (0, 0, 0),
                -1
            )

            cv2.putText(
                frame,
                f"Gesto: {label}",
                (15, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                status,
                (15, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2
            )

            cv2.putText(
                frame,
                f"Frames: {len(sequence)}",
                (15, 105),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                (
                    "ESPACO = iniciar/parar | "
                    "Q = cancelar"
                ),
                (15, 135),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (220, 220, 220),
                1
            )

            cv2.imshow(
                "IA Libras V4 - Dinamico",
                frame
            )

            key = cv2.waitKey(1) & 0xFF

            if key == ord(" "):

                if not recording:

                    recording = True
                    sequence = []

                    print(
                        "● Gravação iniciada."
                    )

                else:

                    recording = False

                    print(
                        "■ Gravação finalizada."
                    )

                    break

            elif key == ord("q"):

                sequence = []

                break

        cap.release()

        cv2.destroyAllWindows()

        if len(sequence) < 10:

            print(
                "✗ Gravação muito curta."
            )

            return

        sequence = (
            self.normalize_sequence(
                sequence
            )
        )

        number = self.next_file_number(
            label_dir
        )

        filename = os.path.join(
            label_dir,
            f"gesto_{number:03d}.npy"
        )

        np.save(
            filename,
            sequence
        )

        print()
        print(
            "✓ Gesto salvo."
        )

        print(
            f"✓ Classe: {label}"
        )

        print(
            f"✓ Arquivo: {filename}"
        )

    # ========================================================
    # LISTAR DATASET
    # ========================================================

    def list_dataset(self):

        print()
        print("=" * 65)
        print(
            "                  DATASET LIBRAS"
        )
        print("=" * 65)

        total_static = 0
        total_dynamic = 0

        print()
        print(
            "LETRAS / NÚMEROS:"
        )

        if os.path.exists(
            STATIC_DATASET_DIR
        ):

            for label in sorted(
                os.listdir(
                    STATIC_DATASET_DIR
                )
            ):

                path = os.path.join(
                    STATIC_DATASET_DIR,
                    label
                )

                if not os.path.isdir(path):
                    continue

                files = [
                    f
                    for f in os.listdir(path)
                    if f.endswith(".npy")
                ]

                if files:

                    print(
                        f"  {label:<15}"
                        f"{len(files)} arquivos"
                    )

                    total_static += len(files)

        print()
        print(
            "GESTOS / PALAVRAS:"
        )

        if os.path.exists(
            DYNAMIC_DATASET_DIR
        ):

            for label in sorted(
                os.listdir(
                    DYNAMIC_DATASET_DIR
                )
            ):

                path = os.path.join(
                    DYNAMIC_DATASET_DIR,
                    label
                )

                if not os.path.isdir(path):
                    continue

                files = [
                    f
                    for f in os.listdir(path)
                    if f.endswith(".npy")
                ]

                if files:

                    print(
                        f"  {label:<15}"
                        f"{len(files)} arquivos"
                    )

                    total_dynamic += len(files)

        print()
        print(
            f"Total estáticos: "
            f"{total_static}"
        )

        print(
            f"Total dinâmicos: "
            f"{total_dynamic}"
        )

    # ========================================================
    # CARREGAR DATASET ESTÁTICO
    # ========================================================

    def load_static_dataset(self):

        X = []
        y = []

        if not os.path.exists(
            STATIC_DATASET_DIR
        ):

            return None, None

        for label in sorted(
            os.listdir(
                STATIC_DATASET_DIR
            )
        ):

            label_dir = os.path.join(
                STATIC_DATASET_DIR,
                label
            )

            if not os.path.isdir(
                label_dir
            ):

                continue

            for filename in os.listdir(
                label_dir
            ):

                if not filename.endswith(
                    ".npy"
                ):

                    continue

                filepath = os.path.join(
                    label_dir,
                    filename
                )

                try:

                    data = np.load(
                        filepath
                    )

                    if data.ndim != 2:
                        continue

                    if (
                        data.shape[1]
                        != TOTAL_FEATURES
                    ):

                        continue

                    for sample in data:

                        X.append(sample)
                        y.append(label)

                except Exception as error:

                    print(
                        f"Erro em {filepath}: "
                        f"{error}"
                    )

        if not X:

            return None, None

        return (
            np.array(
                X,
                dtype=np.float32
            ),
            np.array(y)
        )

    # ========================================================
    # CARREGAR DATASET DINÂMICO
    # ========================================================

    def load_dynamic_dataset(self):

        X = []
        y = []

        if not os.path.exists(
            DYNAMIC_DATASET_DIR
        ):

            return None, None

        for label in sorted(
            os.listdir(
                DYNAMIC_DATASET_DIR
            )
        ):

            label_dir = os.path.join(
                DYNAMIC_DATASET_DIR,
                label
            )

            if not os.path.isdir(
                label_dir
            ):

                continue

            for filename in os.listdir(
                label_dir
            ):

                if not filename.endswith(
                    ".npy"
                ):

                    continue

                filepath = os.path.join(
                    label_dir,
                    filename
                )

                try:

                    sequence = np.load(
                        filepath
                    )

                    if sequence.shape == (
                        SEQUENCE_LENGTH,
                        TOTAL_FEATURES
                    ):

                        X.append(sequence)
                        y.append(label)

                except Exception as error:

                    print(
                        f"Erro em {filepath}: "
                        f"{error}"
                    )

        if not X:

            return None, None

        return (
            np.array(
                X,
                dtype=np.float32
            ),
            np.array(y)
        )

    # ========================================================
    # TREINAR MODELO ESTÁTICO
    # ========================================================

    def train_static_model(self):

        print()
        print("=" * 65)
        print(
            "       TREINAMENTO - LETRAS / NÚMEROS"
        )
        print("=" * 65)

        X, y = (
            self.load_static_dataset()
        )

        if X is None:

            print()
            print(
                "✗ Nenhum exemplo estático encontrado."
            )

            print(
                "Grave letras ou números primeiro."
            )

            return

        labels = sorted(
            np.unique(y)
        )

        print()
        print(
            f"Exemplos: {len(X)}"
        )

        print(
            f"Classes: {len(labels)}"
        )

        if len(labels) < 2:

            print(
                "✗ É necessário ter pelo menos "
                "2 classes."
            )

            return

        encoder = LabelEncoder()

        y_encoded = encoder.fit_transform(
            y
        )

        try:

            X_train, X_test, y_train, y_test = (
                train_test_split(
                    X,
                    y_encoded,
                    test_size=0.20,
                    random_state=42,
                    stratify=y_encoded
                )
            )

        except ValueError:

            print(
                "✗ Poucos exemplos para "
                "separar treino/teste."
            )

            print(
                "Recomendo pelo menos "
                "20-40 exemplos por classe."
            )

            return

        print()
        print(
            "Treinando RandomForest..."
        )

        model = RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            n_jobs=-1,
            class_weight="balanced"
        )

        model.fit(
            X_train,
            y_train
        )

        prediction = model.predict(
            X_test
        )

        accuracy = accuracy_score(
            y_test,
            prediction
        )

        self.static_model = model

        self.static_label_encoder = (
            encoder
        )

        joblib.dump(
            model,
            STATIC_MODEL_FILE
        )

        with open(
            STATIC_LABELS_FILE,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                list(
                    encoder.classes_
                ),
                file,
                ensure_ascii=False,
                indent=4
            )

        print()
        print("=" * 65)

        print(
            f"Acurácia de teste: "
            f"{accuracy * 100:.2f}%"
        )

        print("=" * 65)

        print(
            "✓ Modelo estático salvo."
        )

    # ========================================================
    # CRIAR MODELO GRU
    # ========================================================

    def create_dynamic_model(
        self,
        number_of_classes
    ):

        model = Sequential([

            Masking(
                mask_value=0.0,
                input_shape=(
                    SEQUENCE_LENGTH,
                    TOTAL_FEATURES
                )
            ),

            GRU(
                96,
                return_sequences=True
            ),

            Dropout(0.25),

            GRU(48),

            Dropout(0.25),

            Dense(
                48,
                activation="relu"
            ),

            Dropout(0.20),

            Dense(
                number_of_classes,
                activation="softmax"
            )
        ])

        model.compile(
            optimizer="adam",
            loss="categorical_crossentropy",
            metrics=["accuracy"]
        )

        return model

    # ========================================================
    # TREINAR MODELO DINÂMICO
    # ========================================================

    def train_dynamic_model(self):

        print()
        print("=" * 65)
        print(
            "             TREINAMENTO - GESTOS"
        )
        print("=" * 65)

        X, y = (
            self.load_dynamic_dataset()
        )

        if X is None:

            print()
            print(
                "✗ Nenhum gesto dinâmico encontrado."
            )

            print(
                "Grave alguns gestos primeiro."
            )

            return

        labels = sorted(
            np.unique(y)
        )

        print()
        print(
            f"Gravações: {len(X)}"
        )

        print(
            f"Classes: {len(labels)}"
        )

        if len(labels) < 2:

            print(
                "✗ É necessário ter pelo menos "
                "2 gestos."
            )

            return

        encoder = LabelEncoder()

        y_encoded = encoder.fit_transform(
            y
        )

        y_categorical = to_categorical(
            y_encoded,
            num_classes=len(labels)
        )

        try:

            X_train, X_test, y_train, y_test = (
                train_test_split(
                    X,
                    y_categorical,
                    test_size=0.20,
                    random_state=42,
                    stratify=y_encoded
                )
            )

        except ValueError:

            print(
                "✗ Não há exemplos suficientes."
            )

            print(
                "Recomendo pelo menos "
                "5 gravações por gesto."
            )

            return

        model = self.create_dynamic_model(
            len(labels)
        )

        print()
        print(
            "Iniciando treinamento..."
        )

        model.fit(
            X_train,
            y_train,
            validation_data=(
                X_test,
                y_test
            ),
            epochs=40,
            batch_size=16,
            verbose=1
        )

        loss, accuracy = model.evaluate(
            X_test,
            y_test,
            verbose=0
        )

        model.save(
            DYNAMIC_MODEL_FILE
        )

        with open(
            DYNAMIC_LABELS_FILE,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                list(
                    encoder.classes_
                ),
                file,
                ensure_ascii=False,
                indent=4
            )

        self.dynamic_model = model

        self.dynamic_label_encoder = (
            encoder
        )

        print()
        print("=" * 65)

        print(
            f"Acurácia de teste: "
            f"{accuracy * 100:.2f}%"
        )

        print("=" * 65)

        print(
            "✓ Modelo dinâmico salvo."
        )

    # ========================================================
    # TREINAR OS DOIS MODELOS
    # ========================================================

    def train_models(self):

        print()
        print("=" * 65)
        print(
            "                 TREINAR IA V4"
        )
        print("=" * 65)

        self.train_static_model()

        self.train_dynamic_model()

        print()
        print(
            "✓ Treinamento finalizado."
        )

    # ========================================================
    # CARREGAR MODELO ESTÁTICO
    # ========================================================

    def load_static_model(self):

        if self.static_model is not None:

            return True

        if not os.path.exists(
            STATIC_MODEL_FILE
        ):

            return False

        if not os.path.exists(
            STATIC_LABELS_FILE
        ):

            return False

        try:

            self.static_model = (
                joblib.load(
                    STATIC_MODEL_FILE
                )
            )

            with open(
                STATIC_LABELS_FILE,
                "r",
                encoding="utf-8"
            ) as file:

                labels = json.load(
                    file
                )

            self.static_label_encoder = (
                LabelEncoder()
            )

            self.static_label_encoder.fit(
                labels
            )

            return True

        except Exception as error:

            print(
                "Erro ao carregar modelo "
                f"estático: {error}"
            )

            return False

    # ========================================================
    # CARREGAR MODELO DINÂMICO
    # ========================================================

    def load_dynamic_model(self):

        if self.dynamic_model is not None:

            return True

        if not os.path.exists(
            DYNAMIC_MODEL_FILE
        ):

            return False

        if not os.path.exists(
            DYNAMIC_LABELS_FILE
        ):

            return False

        try:

            self.dynamic_model = (
                load_model(
                    DYNAMIC_MODEL_FILE
                )
            )

            with open(
                DYNAMIC_LABELS_FILE,
                "r",
                encoding="utf-8"
            ) as file:

                labels = json.load(
                    file
                )

            self.dynamic_label_encoder = (
                LabelEncoder()
            )

            self.dynamic_label_encoder.fit(
                labels
            )

            return True

        except Exception as error:

            print(
                "Erro ao carregar modelo "
                f"dinâmico: {error}"
            )

            return False

    # ========================================================
    # CALCULAR MOVIMENTO
    # ========================================================

    def calculate_motion(
        self,
        sequence
    ):

        if len(sequence) < 2:

            return 0.0

        current = sequence[-1]

        previous = sequence[-2]

        difference = np.abs(
            current - previous
        )

        return float(
            np.mean(difference)
        )

    # ========================================================
    # PREVER ESTÁTICO
    # ========================================================

    def predict_static(
        self,
        features
    ):

        if not self.load_static_model():

            return (
                "Sem modelo",
                0.0
            )

        try:

            probabilities = (
                self.static_model
                .predict_proba(
                    features.reshape(
                        1,
                        -1
                    )
                )[0]
            )

            index = np.argmax(
                probabilities
            )

            confidence = (
                probabilities[index]
                * 100
            )

            label = (
                self.static_label_encoder
                .inverse_transform(
                    [index]
                )[0]
            )

            return (
                label,
                confidence
            )

        except Exception:

            return (
                "Não reconhecido",
                0.0
            )

    # ========================================================
    # PREVER DINÂMICO
    # ========================================================

    def predict_dynamic(
        self,
        sequence
    ):

        if not self.load_dynamic_model():

            return (
                "Sem modelo",
                0.0
            )

        try:

            input_sequence = (
                self.normalize_sequence(
                    sequence
                )
            )

            input_sequence = (
                input_sequence.reshape(
                    1,
                    SEQUENCE_LENGTH,
                    TOTAL_FEATURES
                )
            )

            prediction = (
                self.dynamic_model
                .predict(
                    input_sequence,
                    verbose=0
                )[0]
            )

            index = np.argmax(
                prediction
            )

            confidence = (
                prediction[index]
                * 100
            )

            label = (
                self.dynamic_label_encoder
                .inverse_transform(
                    [index]
                )[0]
            )

            return (
                label,
                confidence
            )

        except Exception:

            return (
                "Não reconhecido",
                0.0
            )

    # ========================================================
    # RECONHECIMENTO EM TEMPO REAL
    # ========================================================

    def realtime(self):

        print()
        print("=" * 65)
        print(
            "          RECONHECIMENTO EM TEMPO REAL"
        )
        print("=" * 65)

        static_available = (
            self.load_static_model()
        )

        dynamic_available = (
            self.load_dynamic_model()
        )

        if (
            not static_available
            and
            not dynamic_available
        ):

            print()
            print(
                "✗ Nenhum modelo treinado."
            )

            print(
                "Treine a IA primeiro."
            )

            return

        cap = self.open_camera()

        if cap is None:

            print(
                "✗ Não foi possível abrir "
                "a câmera."
            )

            return

        print()
        print(
            "✓ Câmera iniciada."
        )

        print(
            "O sistema escolherá automaticamente "
            "entre estático e dinâmico."
        )

        print(
            "Q = sair"
        )

        sequence = []

        camera_frame = 0

        prediction_counter = 0

        last_label = (
            "Aguardando..."
        )

        last_confidence = 0.0

        last_mode = "Nenhum"

        fps = 0

        fps_counter = 0

        fps_start = time.time()

        self.reset_timestamp()

        self.last_results = None

        while True:

            ret, frame = cap.read()

            if not ret:
                break

            camera_frame += 1

            frame = cv2.flip(
                frame,
                1
            )

            if (
                camera_frame
                % PROCESS_EVERY
                == 0
            ):

                results = (
                    self.process_frame(
                        frame
                    )
                )

                hand_detected = (
                    self.has_hand(
                        results
                    )
                )

                if hand_detected:

                    features = (
                        self.extract_features(
                            results
                        )
                    )

                    sequence.append(
                        features
                    )

                    if (
                        len(sequence)
                        > SEQUENCE_LENGTH
                    ):

                        sequence.pop(0)

                    prediction_counter += 1

                    if (
                        prediction_counter
                        >= PREDICT_EVERY
                    ):

                        prediction_counter = 0

                        motion = (
                            self.calculate_motion(
                                sequence
                            )
                        )

                        # ------------------------------------
                        # GESTO DINÂMICO
                        # ------------------------------------

                        if (
                            motion
                            >= MOTION_THRESHOLD
                            and
                            len(sequence)
                            >= SEQUENCE_LENGTH
                            and
                            dynamic_available
                        ):

                            label, confidence = (
                                self.predict_dynamic(
                                    sequence
                                )
                            )

                            if (
                                confidence
                                >=
                                MIN_CONFIDENCE
                                * 100
                            ):

                                last_label = label

                                last_confidence = (
                                    confidence
                                )

                                last_mode = (
                                    "DINÂMICO"
                                )

                        # ------------------------------------
                        # LETRA / NÚMERO
                        # ------------------------------------

                        elif static_available:

                            label, confidence = (
                                self.predict_static(
                                    features
                                )
                            )

                            if (
                                confidence
                                >=
                                MIN_CONFIDENCE
                                * 100
                            ):

                                last_label = label

                                last_confidence = (
                                    confidence
                                )

                                last_mode = (
                                    "ESTÁTICO"
                                )

            else:

                results = (
                    self.last_results
                )

            self.draw_hands(
                frame,
                self.last_results
            )

            # ------------------------------------------------
            # FPS
            # ------------------------------------------------

            fps_counter += 1

            elapsed = (
                time.time()
                - fps_start
            )

            if elapsed >= 1.0:

                fps = (
                    fps_counter
                    / elapsed
                )

                fps_counter = 0

                fps_start = (
                    time.time()
                )

            # ------------------------------------------------
            # INTERFACE
            # ------------------------------------------------

            cv2.rectangle(
                frame,
                (0, 0),
                (CAMERA_WIDTH, 190),
                (0, 0, 0),
                -1
            )

            cv2.putText(
                frame,
                f"Sinal: {last_label}",
                (15, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                f"Tipo: {last_mode}",
                (15, 75),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                (
                    f"Confianca: "
                    f"{last_confidence:.1f}%"
                ),
                (15, 105),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                f"FPS: {fps:.1f}",
                (15, 135),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                (
                    f"Frames: "
                    f"{len(sequence)}/"
                    f"{SEQUENCE_LENGTH}"
                ),
                (15, 160),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (220, 220, 220),
                1
            )

            cv2.putText(
                frame,
                "Q = sair",
                (15, 182),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (200, 200, 200),
                1
            )

            cv2.imshow(
                "IA Libras V4",
                frame
            )

            key = (
                cv2.waitKey(1)
                & 0xFF
            )

            if key == ord("q"):

                break

        cap.release()

        cv2.destroyAllWindows()

    # ========================================================
    # EXCLUIR CLASSE
    # ========================================================

    def delete_gesture(self):

        print()
        print("=" * 65)
        print(
            "                  EXCLUIR CLASSE"
        )
        print("=" * 65)

        print()
        print("1. Estático")
        print("2. Dinâmico")

        choice = input(
            "Escolha: "
        ).strip()

        if choice == "1":

            base_dir = (
                STATIC_DATASET_DIR
            )

            kind = "estático"

        elif choice == "2":

            base_dir = (
                DYNAMIC_DATASET_DIR
            )

            kind = "dinâmico"

        else:

            print(
                "✗ Opção inválida."
            )

            return

        if not os.path.exists(
            base_dir
        ):

            print(
                "✗ Nenhuma classe encontrada."
            )

            return

        print()
        print(
            "Classes:"
        )

        for label in sorted(
            os.listdir(base_dir)
        ):

            path = os.path.join(
                base_dir,
                label
            )

            if os.path.isdir(path):

                print(
                    f" - {label}"
                )

        print()

        label = input(
            f"Digite o {kind} "
            "para excluir: "
        ).strip().upper()

        path = os.path.join(
            base_dir,
            label
        )

        if not os.path.exists(path):

            print(
                "✗ Classe não encontrada."
            )

            return

        confirmation = input(
            (
                f"Excluir TODOS os "
                f"exemplos de {label}? "
                "(S/N): "
            )
        ).strip().upper()

        if confirmation != "S":

            print(
                "Operação cancelada."
            )

            return

        shutil.rmtree(path)

        print(
            f"✓ Classe {label} excluída."
        )

    # ========================================================
    # FECHAR
    # ========================================================

    def close(self):

        try:

            self.hand_landmarker.close()

        except Exception:

            pass


# ============================================================
# MENU PRINCIPAL
# ============================================================

def main():

    ai = LibrasAI()

    try:

        while True:

            print()
            print("=" * 65)
            print(
                "                       IA LIBRAS V4"
            )
            print("=" * 65)

            print()

            print(
                "1. Gravar letra/número estático"
            )

            print(
                "2. Gravar gesto/palavra dinâmico"
            )

            print(
                "3. Ver dados cadastrados"
            )

            print(
                "4. Treinar IA"
            )

            print(
                "5. Reconhecimento em tempo real"
            )

            print(
                "6. Excluir classe"
            )

            print(
                "7. Sair"
            )

            print()
            print("=" * 65)

            choice = input(
                "Escolha uma opção: "
            ).strip()

            if choice == "1":

                ai.record_static()

            elif choice == "2":

                ai.record_dynamic()

            elif choice == "3":

                ai.list_dataset()

            elif choice == "4":

                ai.train_models()

            elif choice == "5":

                ai.realtime()

            elif choice == "6":

                ai.delete_gesture()

            elif choice == "7":

                print()
                print(
                    "Encerrando IA Libras..."
                )

                break

            else:

                print()
                print(
                    "✗ Opção inválida."
                )

    finally:

        ai.close()


# ============================================================
# EXECUÇÃO
# ============================================================

if __name__ == "__main__":
    main()