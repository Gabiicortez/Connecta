#============================================================

#IA LIBRAS - V3

#RECONHECIMENTO DE LETRAS, GESTOS E PALAVRAS

#============================================================



#MediaPipe 1.0.1

#OpenCV

#TensorFlow / GRU



#V3 - OTIMIZAÇÃO DE CÂMERA



#Melhorias:

#- Câmera em 640x480

#- MediaPipe analisa em 320x240

#- MediaPipe não roda em todos os frames

#- GRU não roda em todos os frames

#- Buffer da câmera reduzido

#- Suporte para duas mãos

#- Gravação de movimentos

#- Treinamento de palavras/gestos

#- Reconhecimento em tempo real

#- FPS na tela



#============================================================

import cv2
import mediapipe as mp
import numpy as np
import os
import json
import time
import urllib.request
import shutil

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import GRU, Dense, Dropout, Masking
from tensorflow.keras.utils import to_categorical

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

#============================================================

#CONFIGURAÇÕES

#============================================================

MODEL_PATH = "hand_landmarker.task"

MODEL_URL = (
"https://storage.googleapis.com/mediapipe-models/"
"hand_landmarker/hand_landmarker/float16/1/"
"hand_landmarker.task"
)

DATASET_DIR = "dataset_libras"

MODEL_FILE = "libras_gru.keras"

LABELS_FILE = "libras_labels.json"

#============================================================

#CONFIGURAÇÃO DA CÂMERA

#============================================================

CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 30

#============================================================

#CONFIGURAÇÃO DO PROCESSAMENTO

#============================================================

#MediaPipe será executado somente a cada 3 frames.



#1 = mais preciso, mais pesado

#2 = equilíbrio

#3 = mais leve

#4 = ainda mais leve



PROCESS_EVERY = 3

#Resolução usada SOMENTE pelo MediaPipe.



#A janela continua 640x480.

PROCESS_WIDTH = 320
PROCESS_HEIGHT = 240

#A GRU será executada a cada 5 análises do MediaPipe.

PREDICT_EVERY = 5

#============================================================

#SEQUÊNCIA

#============================================================

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

#============================================================

#CONFIANÇA

#============================================================

MIN_CONFIDENCE = 0.60

#============================================================

#CLASSE PRINCIPAL

#============================================================

class LibrasAI:

        def __init__(self):

            print()
            print("=" * 65)
            print("                 IA LIBRAS V3")
            print("=" * 65)

            print()
            print("Inicializando sistema...")

            # ----------------------------------------------------
            # Modelo MediaPipe
            # ----------------------------------------------------

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
            # Variáveis
            # ----------------------------------------------------

            self.model = None

            self.label_encoder = None

            self.timestamp = 0

            # Último resultado do MediaPipe
            self.last_results = None

            # ----------------------------------------------------
            # Dataset
            # ----------------------------------------------------

            os.makedirs(
                DATASET_DIR,
                exist_ok=True
            )

            print()
            print("✓ MediaPipe inicializado.")
            print(
                f"✓ Câmera: "
                f"{CAMERA_WIDTH}x{CAMERA_HEIGHT}"
            )
            print(
                f"✓ Análise MediaPipe: "
                f"{PROCESS_WIDTH}x{PROCESS_HEIGHT}"
            )
            print(
                f"✓ MediaPipe a cada "
                f"{PROCESS_EVERY} frames"
            )
            print(
                f"✓ GRU a cada "
                f"{PREDICT_EVERY} análises"
            )
            print(
                f"✓ Sequência: "
                f"{SEQUENCE_LENGTH} frames"
            )
            print(
                f"✓ Features/frame: "
                f"{TOTAL_FEATURES}"
            )

            print()
            print("=" * 65)


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

            # ----------------------------------------------------
            # Primeiro tenta DirectShow
            # ----------------------------------------------------

            camera = cv2.VideoCapture(
                0,
                cv2.CAP_DSHOW
            )

            # ----------------------------------------------------
            # Se falhar, tenta padrão
            # ----------------------------------------------------

            if not camera.isOpened():

                camera.release()

                camera = cv2.VideoCapture(0)


            if not camera.isOpened():

                return None


            # ----------------------------------------------------
            # Resolução
            # ----------------------------------------------------

            camera.set(
                cv2.CAP_PROP_FRAME_WIDTH,
                CAMERA_WIDTH
            )

            camera.set(
                cv2.CAP_PROP_FRAME_HEIGHT,
                CAMERA_HEIGHT
            )


            # ----------------------------------------------------
            # FPS
            # ----------------------------------------------------

            camera.set(
                cv2.CAP_PROP_FPS,
                CAMERA_FPS
            )


            # ----------------------------------------------------
            # Buffer pequeno
            #
            # Evita acumular frames antigos.
            # ----------------------------------------------------

            camera.set(
                cv2.CAP_PROP_BUFFERSIZE,
                1
            )


            # ----------------------------------------------------
            # MJPG
            # ----------------------------------------------------

            camera.set(
                cv2.CAP_PROP_FOURCC,
                cv2.VideoWriter_fourcc(
                    *"MJPG"
                )
            )


            return camera


        # ========================================================
        # TIMESTAMP MEDIA PIPE
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
        # PROCESSAR FRAME NO MEDIAPIPE
        # ========================================================

        def process_frame(self, frame):

            # ----------------------------------------------------
            # Reduz resolução somente para o MediaPipe
            # ----------------------------------------------------

            small_frame = cv2.resize(

                frame,

                (
                    PROCESS_WIDTH,
                    PROCESS_HEIGHT
                ),

                interpolation=cv2.INTER_AREA
            )


            # ----------------------------------------------------
            # BGR -> RGB
            # ----------------------------------------------------

            rgb = cv2.cvtColor(

                small_frame,

                cv2.COLOR_BGR2RGB
            )


            # ----------------------------------------------------
            # MediaPipe Image
            # ----------------------------------------------------

            mp_image = mp.Image(

                image_format=mp.ImageFormat.SRGB,

                data=rgb
            )


            # ----------------------------------------------------
            # Timestamp
            # ----------------------------------------------------

            timestamp = self.get_timestamp()


            # ----------------------------------------------------
            # MediaPipe
            # ----------------------------------------------------

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
        # EXTRAIR DUAS MÃOS
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


            # ----------------------------------------------------
            # Nenhuma mão
            # ----------------------------------------------------

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


            # ----------------------------------------------------
            # Percorre as mãos
            # ----------------------------------------------------

            for index, hand_landmarks in enumerate(

                results.hand_landmarks

            ):

                features = self.extract_hand(

                    hand_landmarks
                )


                # ------------------------------------------------
                # Identificação esquerda/direita
                # ------------------------------------------------

                try:

                    handedness = (

                        results.handedness[index][0]
                    )


                    category = (

                        handedness.category_name
                        .lower()
                    )


                except Exception:

                    category = ""


                # ------------------------------------------------
                # Mão esquerda
                # ------------------------------------------------

                if category == "left":

                    left_hand = features


                # ------------------------------------------------
                # Mão direita
                # ------------------------------------------------

                elif category == "right":

                    right_hand = features


                # ------------------------------------------------
                # Caso não identifique
                # ------------------------------------------------

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
        # DESENHAR MÃOS
        # ========================================================

        def draw_hands(self, frame, results):

            if results is None:

                return


            if not results.hand_landmarks:

                return


            height, width, _ = frame.shape


            # ----------------------------------------------------
            # Conexões dos dedos
            # ----------------------------------------------------

            connections = [

                # Polegar
                (0, 1),
                (1, 2),
                (2, 3),
                (3, 4),

                # Indicador
                (0, 5),
                (5, 6),
                (6, 7),
                (7, 8),

                # Médio
                (0, 9),
                (9, 10),
                (10, 11),
                (11, 12),

                # Anelar
                (0, 13),
                (13, 14),
                (14, 15),
                (15, 16),

                # Mindinho
                (0, 17),
                (17, 18),
                (18, 19),
                (19, 20),

                # Palma
                (5, 9),
                (9, 13),
                (13, 17)
            ]


            # ----------------------------------------------------
            # Desenha cada mão
            # ----------------------------------------------------

            for hand in results.hand_landmarks:

                points = []


                # -----------------------------------------------
                # Pontos
                # -----------------------------------------------

                for landmark in hand:

                    x = int(

                        landmark.x
                        * width
                    )

                    y = int(

                        landmark.y
                        * height
                    )


                    points.append(

                        (
                            x,
                            y
                        )
                    )


                # -----------------------------------------------
                # Linhas
                # -----------------------------------------------

                for start, end in connections:

                    cv2.line(

                        frame,

                        points[start],

                        points[end],

                        (0, 255, 0),

                        2
                    )


                # -----------------------------------------------
                # Pontos
                # -----------------------------------------------

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

        def normalize_sequence(self, sequence):

            sequence = np.array(

                sequence,

                dtype=np.float32
            )


            # ----------------------------------------------------
            # Menos frames
            # ----------------------------------------------------

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


            # ----------------------------------------------------
            # Mais frames
            # ----------------------------------------------------

            elif len(sequence) > SEQUENCE_LENGTH:

                indexes = np.linspace(

                    0,

                    len(sequence) - 1,

                    SEQUENCE_LENGTH
                ).astype(int)


                sequence = sequence[indexes]


            return sequence


        # ========================================================
        # VERIFICAR SE HÁ MÃO
        # ========================================================

        def has_hand(self, results):

            if results is None:

                return False


            if not results.hand_landmarks:

                return False


            return True


        # ========================================================
        # GRAVAR GESTO
        # ========================================================

        def record_gesture(self):

            print()
            print("=" * 65)
            print("                 GRAVAR GESTO")
            print("=" * 65)

            print()
            print("Exemplos de nomes:")
            print("OI")
            print("OBRIGADO")
            print("BOM_DIA")
            print("TUDO_BEM")
            print("SIM")
            print("NAO")
            print("A")
            print("B")
            print()


            label = input(

                "Digite o nome do gesto: "

            ).strip().upper()


            if not label:

                print(
                    "✗ Nome inválido."
                )

                return


            # ----------------------------------------------------
            # Pasta da classe
            # ----------------------------------------------------

            label_dir = os.path.join(

                DATASET_DIR,

                label
            )


            os.makedirs(

                label_dir,

                exist_ok=True
            )


            # ----------------------------------------------------
            # Número do exemplo
            # ----------------------------------------------------

            existing = [

                f

                for f in os.listdir(

                    label_dir

                )

                if f.endswith(".npy")
            ]


            number = len(existing) + 1


            print()
            print("=" * 65)
            print("COMO GRAVAR")
            print("=" * 65)

            print()
            print(
                "1. Pressione ESPAÇO para começar."
            )

            print(
                "2. Faça o sinal completo."
            )

            print(
                "3. Pressione ESPAÇO para terminar."
            )

            print(
                "4. Pressione Q para cancelar."
            )

            print()
            print(
                "Dica: faça o movimento naturalmente."
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

            processed_frame = 0

            self.reset_timestamp()


            while True:

                ret, frame = cap.read()


                if not ret:

                    print(
                        "✗ Erro ao capturar câmera."
                    )

                    break


                camera_frame += 1


                # ------------------------------------------------
                # Espelha
                # ------------------------------------------------

                frame = cv2.flip(

                    frame,

                    1
                )


                # ------------------------------------------------
                # MediaPipe somente a cada N frames
                # ------------------------------------------------

                if (

                    camera_frame
                    % PROCESS_EVERY
                    == 0
                ):

                    results = self.process_frame(

                        frame
                    )

                    processed_frame += 1

                else:

                    results = self.last_results


                # ------------------------------------------------
                # Desenhar
                # ------------------------------------------------

                self.draw_hands(

                    frame,

                    results
                )


                # ------------------------------------------------
                # Gravar somente nos frames
                # processados pelo MediaPipe
                # ------------------------------------------------

                if recording:

                    if (

                        camera_frame
                        % PROCESS_EVERY
                        == 0
                    ):

                        features = (

                            self.extract_features(

                                results
                            )
                        )


                        sequence.append(

                            features
                        )


                # ------------------------------------------------
                # Status
                # ------------------------------------------------

                if recording:

                    status = "GRAVANDO"

                    status_color = (

                        0,
                        0,
                        255
                    )

                else:

                    status = "AGUARDANDO"

                    status_color = (

                        0,
                        255,
                        255
                    )


                # ------------------------------------------------
                # Interface
                # ------------------------------------------------

                cv2.rectangle(

                    frame,

                    (
                        0,
                        0
                    ),

                    (
                        CAMERA_WIDTH,
                        155
                    ),

                    (
                        0,
                        0,
                        0
                    ),

                    -1
                )


                cv2.putText(

                    frame,

                    f"Gesto: {label}",

                    (
                        15,
                        35
                    ),

                    cv2.FONT_HERSHEY_SIMPLEX,

                    0.8,

                    (
                        255,
                        255,
                        255
                    ),

                    2
                )


                cv2.putText(

                    frame,

                    status,

                    (
                        15,
                        70
                    ),

                    cv2.FONT_HERSHEY_SIMPLEX,

                    0.8,

                    status_color,

                    2
                )


                cv2.putText(

                    frame,

                    f"Frames: {len(sequence)}",

                    (
                        15,
                        105
                    ),

                    cv2.FONT_HERSHEY_SIMPLEX,

                    0.6,

                    (
                        255,
                        255,
                        255
                    ),

                    2
                )


                cv2.putText(

                    frame,

                    "ESPACO = iniciar/parar | Q = cancelar",

                    (
                        15,
                        135
                    ),

                    cv2.FONT_HERSHEY_SIMPLEX,

                    0.5,

                    (
                        210,
                        210,
                        210
                    ),

                    1
                )


                cv2.imshow(

                    "IA Libras V3 - Gravacao",

                    frame
                )


                # ------------------------------------------------
                # Teclado
                # ------------------------------------------------

                key = (

                    cv2.waitKey(1)
                    & 0xFF
                )


                # ------------------------------------------------
                # Espaço
                # ------------------------------------------------

                if key == ord(" "):

                    if not recording:

                        recording = True

                        sequence = []

                        print()
                        print(
                            "● GRAVAÇÃO INICIADA"
                        )

                    else:

                        recording = False

                        print()
                        print(
                            "■ GRAVAÇÃO FINALIZADA"
                        )

                        break


                # ------------------------------------------------
                # Q
                # ------------------------------------------------

                elif key == ord("q"):

                    sequence = []

                    print(
                        "Gravação cancelada."
                    )

                    break


            cap.release()

            cv2.destroyAllWindows()


            # ----------------------------------------------------
            # Verificação
            # ----------------------------------------------------

            if len(sequence) < 10:

                print()
                print(
                    "✗ Gravação muito curta."
                )

                print(
                    "Faça o sinal durante mais tempo."
                )

                return


            # ----------------------------------------------------
            # Normalizar
            # ----------------------------------------------------

            sequence = self.normalize_sequence(

                sequence
            )


            # ----------------------------------------------------
            # Salvar
            # ----------------------------------------------------

            filename = os.path.join(

                label_dir,

                f"gesto_{number:03d}.npy"
            )


            np.save(

                filename,

                sequence
            )


            print()
            print("=" * 65)
            print("✓ GESTO SALVO")
            print("=" * 65)

            print(
                f"Gesto: {label}"
            )

            print(
                f"Arquivo: {filename}"
            )

            print(
                f"Frames analisados: "
                f"{len(sequence)}"
            )


        # ========================================================
        # LISTAR DATASET
        # ========================================================

        def list_dataset(self):

            print()
            print("=" * 65)
            print("                DATASET LIBRAS")
            print("=" * 65)

            total = 0

            classes = 0


            if not os.path.exists(

                DATASET_DIR

            ):

                print(
                    "Nenhum gesto cadastrado."
                )

                return


            for label in sorted(

                os.listdir(DATASET_DIR)

            ):

                path = os.path.join(

                    DATASET_DIR,

                    label
                )


                if not os.path.isdir(path):

                    continue


                samples = [

                    f

                    for f in os.listdir(path)

                    if f.endswith(".npy")
                ]


                if samples:

                    classes += 1

                    total += len(samples)

                    print(

                        f"{label:<20} "
                        f"{len(samples)} gravações"
                    )


            print()
            print(
                f"Classes: {classes}"
            )

            print(
                f"Total de gravações: {total}"
            )


        # ========================================================
        # CARREGAR DATASET
        # ========================================================

        def load_dataset(self):

            X = []

            y = []


            if not os.path.exists(

                DATASET_DIR

            ):

                return None, None


            # ----------------------------------------------------
            # Percorrer classes
            # ----------------------------------------------------

            for label in sorted(

                os.listdir(DATASET_DIR)

            ):

                label_dir = os.path.join(

                    DATASET_DIR,

                    label
                )


                if not os.path.isdir(

                    label_dir

                ):

                    continue


                files = [

                    f

                    for f in os.listdir(

                        label_dir

                    )

                    if f.endswith(".npy")
                ]


                # ------------------------------------------------
                # Arquivos
                # ------------------------------------------------

                for filename in files:

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

                            X.append(

                                sequence
                            )

                            y.append(

                                label
                            )


                    except Exception as error:

                        print()

                        print(
                            f"Erro em {filepath}"
                        )

                        print(error)


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
        # CRIAR MODELO GRU
        # ========================================================

        def create_model(

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

                GRU(

                    48
                ),

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

                metrics=[
                    "accuracy"
                ]
            )


            return model


        # ========================================================
        # TREINAR MODELO
        # ========================================================

        def train_model(self):

            print()
            print("=" * 65)
            print("                  TREINAMENTO")
            print("=" * 65)


            X, y = self.load_dataset()


            if X is None:

                print()
                print(
                    "✗ Nenhum dado encontrado."
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
                f"Total de gravações: "
                f"{len(X)}"
            )

            print(
                f"Total de classes: "
                f"{len(labels)}"
            )


            print()
            print("Classes:")


            for label in labels:

                count = np.sum(

                    y == label
                )


                print(

                    f"  {label:<20} "
                    f"{count} exemplos"
                )


            # ----------------------------------------------------
            # Pelo menos duas classes
            # ----------------------------------------------------

            if len(labels) < 2:

                print()
                print(
                    "✗ É necessário ter pelo menos "
                    "2 gestos diferentes."
                )

                return


            # ----------------------------------------------------
            # Encoder
            # ----------------------------------------------------

            self.label_encoder = (

                LabelEncoder()
            )


            y_encoded = (

                self.label_encoder
                .fit_transform(y)
            )


            y_categorical = (

                to_categorical(

                    y_encoded,

                    num_classes=len(labels)
                )
            )


            # ----------------------------------------------------
            # Salvar labels
            # ----------------------------------------------------

            with open(

                LABELS_FILE,

                "w",

                encoding="utf-8"

            ) as file:

                json.dump(

                    list(

                        self.label_encoder.classes_
                    ),

                    file,

                    ensure_ascii=False,

                    indent=4
                )


            # ----------------------------------------------------
            # Separar treinamento/teste
            # ----------------------------------------------------

            try:

                X_train, X_test, y_train, y_test = (

                    train_test_split(

                        X,

                        y_categorical,

                        test_size=0.2,

                        random_state=42,

                        stratify=y_encoded
                    )
                )


            except ValueError:

                print()
                print(
                    "✗ Não há exemplos suficientes "
                    "para dividir o dataset."
                )

                print()
                print(
                    "Recomendo pelo menos "
                    "5 gravações de cada gesto."
                )

                return


            # ----------------------------------------------------
            # Criar modelo
            # ----------------------------------------------------

            self.model = self.create_model(

                len(labels)
            )


            print()
            print("=" * 65)
            print("INICIANDO TREINAMENTO")
            print("=" * 65)

            print()


            # ----------------------------------------------------
            # Treinamento
            # ----------------------------------------------------

            self.model.fit(

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


            # ----------------------------------------------------
            # Avaliação
            # ----------------------------------------------------

            loss, accuracy = (

                self.model.evaluate(

                    X_test,

                    y_test,

                    verbose=0
                )
            )


            print()
            print("=" * 65)

            print(

                f"Acurácia de teste: "
                f"{accuracy * 100:.2f}%"
            )

            print("=" * 65)


            # ----------------------------------------------------
            # Salvar modelo
            # ----------------------------------------------------

            self.model.save(

                MODEL_FILE
            )


            print()
            print(
                f"✓ Modelo salvo em: "
                f"{MODEL_FILE}"
            )

            print(
                f"✓ Labels salvos em: "
                f"{LABELS_FILE}"
            )


        # ========================================================
        # CARREGAR MODELO
        # ========================================================

        def load_trained_model(self):

            if not os.path.exists(

                MODEL_FILE

            ):

                return False


            if not os.path.exists(

                LABELS_FILE

            ):

                return False


            try:

                # -----------------------------------------------
                # Modelo
                # -----------------------------------------------

                self.model = load_model(

                    MODEL_FILE
                )


                # -----------------------------------------------
                # Labels
                # -----------------------------------------------

                with open(

                    LABELS_FILE,

                    "r",

                    encoding="utf-8"

                ) as file:

                    labels = json.load(file)


                self.label_encoder = (

                    LabelEncoder()
                )


                self.label_encoder.fit(

                    labels
                )


                return True


            except Exception as error:

                print()
                print(
                    "✗ Erro ao carregar modelo:"
                )

                print(error)

                return False


        # ========================================================
        # RECONHECIMENTO EM TEMPO REAL
        # ========================================================

        def realtime(self):

            print()
            print("=" * 65)
            print("          RECONHECIMENTO EM TEMPO REAL")
            print("=" * 65)


            # ----------------------------------------------------
            # Carregar modelo
            # ----------------------------------------------------

            if not self.load_trained_model():

                print()
                print(
                    "✗ Modelo treinado não encontrado."
                )

                print(
                    "Escolha a opção 3 para treinar."
                )

                return


            # ----------------------------------------------------
            # Abrir câmera
            # ----------------------------------------------------

            cap = self.open_camera()


            if cap is None:

                print(
                    "✗ Não foi possível abrir a câmera."
                )

                return


            print()
            print(
                "✓ Câmera iniciada."
            )

            print(
                "Q = sair"
            )


            # ----------------------------------------------------
            # Variáveis
            # ----------------------------------------------------

            sequence = []

            camera_frame = 0

            processed_frame = 0

            prediction_counter = 0

            last_label = "Aguardando..."

            last_confidence = 0.0

            last_hand_detected = False


            # FPS
            fps = 0

            fps_counter = 0

            fps_start = time.time()


            self.reset_timestamp()

            self.last_results = None


            # ====================================================
            # LOOP PRINCIPAL
            # ====================================================

            while True:

                # ------------------------------------------------
                # Capturar frame
                # ------------------------------------------------

                ret, frame = cap.read()


                if not ret:

                    print(
                        "✗ Erro ao capturar frame."
                    )

                    break


                camera_frame += 1


                # ------------------------------------------------
                # Espelhar
                # ------------------------------------------------

                frame = cv2.flip(

                    frame,

                    1
                )


                # =================================================
                # MEDIAPIPE
                # =================================================
                #
                # Aqui está uma das principais otimizações:
                #
                # MediaPipe NÃO roda em todos os frames.
                # =================================================

                if (

                    camera_frame
                    % PROCESS_EVERY
                    == 0

                ):

                    results = self.process_frame(

                        frame
                    )


                    processed_frame += 1


                    # ---------------------------------------------
                    # Verificar mão
                    # ---------------------------------------------

                    last_hand_detected = (

                        self.has_hand(

                            results
                        )
                    )


                    # ---------------------------------------------
                    # Extrair landmarks
                    # ---------------------------------------------

                    features = (

                        self.extract_features(

                            results
                        )
                    )


                    sequence.append(

                        features
                    )


                    # ---------------------------------------------
                    # Limitar sequência
                    # ---------------------------------------------

                    if (

                        len(sequence)
                        > SEQUENCE_LENGTH

                    ):

                        sequence.pop(0)


                    prediction_counter += 1


                    # =================================================
                    # PREDIÇÃO GRU
                    # =================================================

                    if (

                        len(sequence)
                        >= SEQUENCE_LENGTH

                        and

                        prediction_counter
                        >= PREDICT_EVERY

                    ):

                        prediction_counter = 0


                        # ---------------------------------------------
                        # Prepara sequência
                        # ---------------------------------------------

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


                        # ---------------------------------------------
                        # TensorFlow
                        # ---------------------------------------------

                        prediction = (

                            self.model.predict(

                                input_sequence,

                                verbose=0
                            )[0]
                        )


                        # ---------------------------------------------
                        # Melhor classe
                        # ---------------------------------------------

                        index = np.argmax(

                            prediction
                        )


                        confidence = (

                            prediction[index]
                            * 100
                        )


                        label = (

                            self.label_encoder
                            .inverse_transform(

                                [index]
                            )[0]
                        )


                        last_confidence = (

                            confidence
                        )


                        # ---------------------------------------------
                        # Confiança
                        # ---------------------------------------------

                        if (

                            confidence
                            >= MIN_CONFIDENCE * 100

                            and

                            last_hand_detected

                        ):

                            last_label = label

                        else:

                            last_label = (
                                "Não reconhecido"
                            )


                # =================================================
                # DESENHAR MÃOS
                # =================================================

                # Usa o último resultado disponível.
                # Isso evita executar MediaPipe novamente.
                self.draw_hands(

                    frame,

                    self.last_results
                )


                # =================================================
                # FPS
                # =================================================

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

                    fps_start = time.time()


                # =================================================
                # INTERFACE
                # =================================================

                cv2.rectangle(

                    frame,

                    (
                        0,
                        0
                    ),

                    (
                        CAMERA_WIDTH,
                        170
                    ),

                    (
                        0,
                        0,
                        0
                    ),

                    -1
                )


                # ------------------------------------------------
                # Sinal
                # ------------------------------------------------

                cv2.putText(

                    frame,

                    f"Sinal: {last_label}",

                    (
                        15,
                        40
                    ),

                    cv2.FONT_HERSHEY_SIMPLEX,

                    1.0,

                    (
                        0,
                        255,
                        0
                    ),

                    2
                )


                # ------------------------------------------------
                # Confiança
                # ------------------------------------------------

                cv2.putText(

                    frame,

                    f"Confianca: "
                    f"{last_confidence:.1f}%",

                    (
                        15,
                        75
                    ),

                    cv2.FONT_HERSHEY_SIMPLEX,

                    0.65,

                    (
                        255,
                        255,
                        255
                    ),

                    2
                )


                # ------------------------------------------------
                # FPS
                # ------------------------------------------------

                cv2.putText(

                    frame,

                    f"FPS: {fps:.1f}",

                    (
                        15,
                        105
                    ),

                    cv2.FONT_HERSHEY_SIMPLEX,

                    0.65,

                    (
                        255,
                        255,
                        0
                    ),

                    2
                )


                # ------------------------------------------------
                # Sequência
                # ------------------------------------------------

                cv2.putText(

                    frame,

                    f"Frames: "
                    f"{len(sequence)}/"
                    f"{SEQUENCE_LENGTH}",

                    (
                        15,
                        135
                    ),

                    cv2.FONT_HERSHEY_SIMPLEX,

                    0.55,

                    (
                        220,
                        220,
                        220
                    ),

                    1
                )


                # ------------------------------------------------
                # Ajuda
                # ------------------------------------------------

                cv2.putText(

                    frame,

                    "Q = sair",

                    (
                        15,
                        160
                    ),

                    cv2.FONT_HERSHEY_SIMPLEX,

                    0.5,

                    (
                        200,
                        200,
                        200
                    ),

                    1
                )


                # =================================================
                # MOSTRAR CÂMERA
                # =================================================

                cv2.imshow(

                    "IA Libras V3",

                    frame
                )


                # =================================================
                # TECLADO
                # =================================================

                key = (

                    cv2.waitKey(1)
                    & 0xFF
                )


                if key == ord("q"):

                    break


            # ----------------------------------------------------
            # Encerrar
            # ----------------------------------------------------

            cap.release()

            cv2.destroyAllWindows()


        # ========================================================
        # EXCLUIR GESTO
        # ========================================================

        def delete_gesture(self):

            self.list_dataset()


            print()


            label = input(

                "Digite o gesto para excluir: "

            ).strip().upper()


            path = os.path.join(

                DATASET_DIR,

                label
            )


            if not os.path.exists(path):

                print(
                    "✗ Gesto não encontrado."
                )

                return


            confirmation = input(

                f"Excluir TODOS os exemplos "
                f"de {label}? (S/N): "

            ).strip().upper()


            if confirmation != "S":

                print(
                    "Operação cancelada."
                )

                return


            shutil.rmtree(path)


            print()

            print(
                f"✓ Gesto {label} excluído."
            )


        # ========================================================
        # FECHAR
        # ========================================================

        def close(self):

            try:

                self.hand_landmarker.close()

            except Exception:

                pass

        #============================================================

#MENU PRINCIPAL

#============================================================

def main():

    ai = LibrasAI()


    try:

            while True:

                    print()
                    print("=" * 65)
                    print("                       IA LIBRAS")
                    print("=" * 65)

                    print()

                    print(
                        "1. Gravar novo gesto"
                    )

                    print(
                        "2. Ver gestos cadastrados"
                    )

                    print(
                        "3. Treinar IA"
                    )

                    print(
                        "4. Reconhecimento em tempo real"
                    )

                    print(
                        "5. Excluir gesto"
                    )

                    print(
                        "6. Sair"
                    )

                    print()
                    print("=" * 65)


                    choice = input(

                        "Escolha uma opção: "

                    ).strip()


                    # ------------------------------------------------
                    # Gravar
                    # ------------------------------------------------

                    if choice == "1":

                            ai.record_gesture()


                        # ------------------------------------------------
                        # Dataset
                        # ------------------------------------------------

                    elif choice == "2":

                            ai.list_dataset()


                        # ------------------------------------------------
                        # Treinar
                        # ------------------------------------------------

                    elif choice == "3":

                            ai.train_model()


                        # ------------------------------------------------
                        # Reconhecimento
                        # ------------------------------------------------

                    elif choice == "4":

                            ai.realtime()


                        # ------------------------------------------------
                        # Excluir
                        # ------------------------------------------------

                    elif choice == "5":

                            ai.delete_gesture()


                        # ------------------------------------------------
                        # Sair
                        # ------------------------------------------------

                    elif choice == "6":

                            print()
                            print(
                                "Encerrando IA Libras..."
                            )

                            break


                        # ------------------------------------------------
                        # Opção inválida
                        # ------------------------------------------------

                    else:

                            print()
                            print(
                                "✗ Opção inválida."
                            )


    finally:

            ai.close()

        #============================================================

#EXECUÇÃO

#============================================================

if __name__ == "__main__":
    main()
