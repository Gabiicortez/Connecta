

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
        print("IA Libras V6.1 Automática | Inicialização sob demanda")
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
