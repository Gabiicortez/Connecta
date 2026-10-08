"""Núcleo independente de câmera/IA para IA Libras V5.

Os arquivos .npy da V4 (126 features por frame, 120 frames para dinâmicos)
continuam utilizáveis. Este módulo não depende do MediaPipe nem do TensorFlow.
"""
from collections import deque
from dataclasses import dataclass
from typing import Optional

import numpy as np

N_HANDS = 2
POINTS = 21
TOTAL_FEATURES = N_HANDS * POINTS * 3
NEW_SEQUENCE_LENGTH = 48
OLD_SEQUENCE_LENGTH = 120
FINGERTIPS = [4, 8, 12, 16, 20]


def _hands(frame):
    frame = np.asarray(frame, dtype=np.float32)
    if frame.shape != (TOTAL_FEATURES,):
        raise ValueError(f"Frame incompatível: {frame.shape}, esperado (126,)")
    return frame.reshape((2, 21, 3))


def _is_present(hand):
    return bool(np.any(hand))


def _palm_size(hand):
    # Medida invariável a posição e relativamente estável à articulação.
    length = np.linalg.norm(hand[9, :2] - hand[0, :2])
    return max(float(length), 0.025)


def normalize_static(frame):
    """Coordenadas locais da mão, mais robustas a distância/posição da câmera."""
    out = np.zeros((2, 21, 3), dtype=np.float32)
    for i, hand in enumerate(_hands(frame)):
        if not _is_present(hand):
            continue
        out[i] = (hand - hand[0]) / _palm_size(hand)
    return np.clip(out, -7, 7).reshape(-1)


def trim_padding(sequence):
    seq = np.asarray(sequence, dtype=np.float32)
    if seq.ndim != 2 or seq.shape[1] != TOTAL_FEATURES:
        raise ValueError(f"Sequência inválida: {seq.shape}")
    present = np.flatnonzero(np.any(seq != 0, axis=1))
    if not len(present):
        return seq[:0]
    return seq[:present[-1] + 1]


def resample_sequence(sequence, length):
    seq = trim_padding(sequence)
    if not len(seq):
        return np.zeros((length, TOTAL_FEATURES), np.float32)
    if len(seq) == 1:
        return np.repeat(seq, length, axis=0)
    xp = np.linspace(0, 1, len(seq))
    target = np.linspace(0, 1, length)
    out = np.empty((length, TOTAL_FEATURES), dtype=np.float32)
    for j in range(TOTAL_FEATURES):
        out[:, j] = np.interp(target, xp, seq[:, j])
    return out


def normalize_dynamic(sequence, length=NEW_SEQUENCE_LENGTH):
    """Mantém deslocamento entre frames, mas remove posição/escala absolutas."""
    seq = trim_padding(sequence)
    if not len(seq):
        return np.zeros((length, TOTAL_FEATURES), dtype=np.float32)
    hands = seq.reshape((-1, 2, 21, 3))
    scales, wrists = [], []
    for frame in hands:
        for hand in frame:
            if _is_present(hand):
                scales.append(_palm_size(hand))
    first = hands[0]
    for hand in first:
        if _is_present(hand):
            wrists.append(hand[0, :2])
    if not wrists:
        for frame in hands:
            wrists = [h[0, :2] for h in frame if _is_present(h)]
            if wrists:
                break
    origin = np.mean(wrists, axis=0) if wrists else np.zeros(2)
    scale = max(float(np.median(scales)), 0.025) if scales else 0.1
    out = np.zeros_like(hands, dtype=np.float32)
    for t, frame in enumerate(hands):
        for i, hand in enumerate(frame):
            if not _is_present(hand):
                continue
            out[t, i, :, :2] = (hand[:, :2] - origin) / scale
            # Z relativo ao pulso: MediaPipe Z relativo à mão (não posição global).
            out[t, i, :, 2] = (hand[:, 2] - hand[0, 2]) / scale
    return np.clip(resample_sequence(out.reshape(-1, TOTAL_FEATURES), length), -12, 12)


def legacy_dynamic_sequence(sequence):
    """Faz inferência em modelos GRU antigos sem alterar os pesos nem as features."""
    seq = trim_padding(sequence)
    if len(seq) <= OLD_SEQUENCE_LENGTH:
        out = np.zeros((OLD_SEQUENCE_LENGTH, TOTAL_FEATURES), dtype=np.float32)
        out[:len(seq)] = seq
        return out
    return resample_sequence(seq, OLD_SEQUENCE_LENGTH)


def motion_score(previous, current):
    """Movimento em unidades do tamanho da palma; ignora escala da webcam."""
    a, b = _hands(previous), _hands(current)
    changes = []
    for old, new in zip(a, b):
        if not (_is_present(old) and _is_present(new)):
            continue
        scale = (_palm_size(old) + _palm_size(new)) / 2
        translate = np.linalg.norm(new[0, :2] - old[0, :2]) / scale
        old_pose = old[FINGERTIPS, :2] - old[0, :2]
        new_pose = new[FINGERTIPS, :2] - new[0, :2]
        shape = np.linalg.norm(new_pose - old_pose, axis=1).mean() / scale
        changes.append(max(float(translate), float(shape) * 0.7))
    return float(np.mean(changes)) if changes else 0.0


@dataclass
class RoutedEvent:
    kind: str  # 'static' ou 'dynamic'
    features: np.ndarray


class GestureRouter:
    """Detector temporal: letra somente após repouso; gesto após fim do movimento.

    Os parâmetros são ajustáveis; a detecção não é um classificador de Libras.
    """
    def __init__(self, start_threshold=0.16, stop_threshold=0.08,
                 hold_seconds=0.50, stop_seconds=0.40, cooldown_seconds=0.85):
        self.start_threshold = float(start_threshold)
        self.stop_threshold = float(stop_threshold)
        self.hold_seconds = hold_seconds
        self.stop_seconds = stop_seconds
        self.cooldown_seconds = cooldown_seconds
        self.reset()

    def reset(self):
        self.previous = None
        self.preroll = deque(maxlen=6)
        self.scores = deque(maxlen=3)
        self.moving = False
        self.motion_frames = []
        self.motion_start = None
        self.quiet_start = None
        self.cooldown_until = 0.0
        self.last_static_event = -1e9
        self.motion_sum = 0.0
        self.motion_peak = 0.0
        self.rise_count = 0
        self.score = 0.0
        self._mode = None

    def feed(self, features: Optional[np.ndarray], now: float, mode="AUTO"):
        if mode != self._mode:
            self.reset()
            self._mode = mode
        if features is None or not np.any(features):
            self.reset()
            self._mode = mode
            return None
        raw = np.asarray(features, dtype=np.float32).copy()
        movement = motion_score(self.previous, raw) if self.previous is not None else 0.0
        self.previous = raw
        self.scores.append(movement)
        self.score = float(np.median(self.scores))
        self.preroll.append(raw)
        if self.score > self.stop_threshold:
            self.quiet_start = None
        elif self.quiet_start is None:
            self.quiet_start = now

        if mode != "ESTATICO":
            if not self.moving:
                if self.score >= self.start_threshold:
                    self.rise_count += 1
                else:
                    self.rise_count = 0
                if self.rise_count >= 2 and now >= self.cooldown_until:
                    self.moving = True
                    self.motion_start = now
                    self.motion_frames = list(self.preroll)
                    self.motion_sum = sum(self.scores)
                    self.motion_peak = self.score
            else:
                self.motion_frames.append(raw)
                self.motion_sum += self.score
                self.motion_peak = max(self.motion_peak, self.score)
                finished = (
                    (self.quiet_start is not None and now - self.quiet_start >= self.stop_seconds)
                    or now - self.motion_start >= 4.5
                )
                if finished:
                    elapsed = now - self.motion_start
                    frames = self.motion_frames
                    is_gesture = (
                        elapsed >= 0.35 and len(frames) >= 6
                        and self.motion_sum >= 0.65
                        and self.motion_peak >= self.start_threshold
                    )
                    self.moving = False
                    self.motion_frames = []
                    self.rise_count = 0
                    self.motion_sum = 0
                    self.cooldown_until = now + self.cooldown_seconds if is_gesture else now + 0.35
                    self.quiet_start = now
                    if is_gesture:
                        return RoutedEvent("dynamic", np.asarray(frames, np.float32))

        if (mode != "DINAMICO" and not self.moving
                and self.quiet_start is not None
                and now - self.quiet_start >= self.hold_seconds
                and now - self.last_static_event >= 0.14
                and now >= self.cooldown_until):
            self.last_static_event = now
            return RoutedEvent("static", raw)
        return None


class StaticConsensus:
    """Não publica uma letra com apenas uma previsão oscilante."""
    def __init__(self, samples=4, votes=3, min_confidence=0.76, min_margin=0.12):
        self.predictions = deque(maxlen=samples)
        self.votes = votes
        self.min_confidence = min_confidence
        self.min_margin = min_margin

    def reset(self):
        self.predictions.clear()

    def update(self, labels, probabilities):
        values = np.asarray(probabilities, dtype=np.float32).ravel()
        if not len(values) or len(values) != len(labels):
            return None
        if not np.all(np.isfinite(values)):
            return None
        self.predictions.append(values)
        if len(self.predictions) < self.votes:
            return None
        indexes = [int(np.argmax(row)) for row in self.predictions]
        avg = np.mean(np.stack(self.predictions), axis=0)
        top = int(np.argmax(avg))
        votes = indexes.count(top)
        sorted_conf = np.sort(avg)
        margin = float(sorted_conf[-1] - sorted_conf[-2]) if len(avg) > 1 else float(avg[top])
        if votes >= self.votes and avg[top] >= self.min_confidence and margin >= self.min_margin:
            return str(labels[top]), float(avg[top])
        return None
