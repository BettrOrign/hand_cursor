import json
import math
import os
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions
from mediapipe.tasks.python.core.base_options import BaseOptions

MODEL_PATH = "hand_landmarker.task"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SETTINGS_PATH = PROJECT_ROOT / "settings.json"

DEFAULT_SETTINGS = {
    "mode": "tilt",
    "sensitivity_x": 2.0,
    "sensitivity_y": 1.5,
    "smooth": 0.3,
    "dead_zone": 0.03,
    "confidence": 0.7,
    "edge_boost": 10.0,
    "pinch_threshold": 0.05,
}


def _load_settings() -> dict:
    try:
        with open(SETTINGS_PATH) as f:
            return {**DEFAULT_SETTINGS, **json.load(f)}
    except (FileNotFoundError, json.JSONDecodeError):
        return dict(DEFAULT_SETTINGS)


# ═══════════════════════════════════════════════════════════════════════════════
#  Режимы управления
# ═══════════════════════════════════════════════════════════════════════════════

def _mode_tilt(detector, frame, lm, s):
    """Наклон костяшек (X) + длина ладони (Y). Калибровка нейтрали."""
    h, w, _ = frame.shape

    # угол линии костяшек 5→17
    angle = math.atan2(lm[17].y - lm[5].y, lm[17].x - lm[5].x)
    # длина запястье→кончик среднего пальца
    hlen = math.hypot(lm[12].x - lm[0].x, lm[12].y - lm[0].y)

    if not detector._calibrated:
        detector._neutral_angle = angle
        detector._neutral_hlen = hlen
        detector._calibrated = True
        if detector._sx is None:
            detector._sx = 0.0
            detector._sy = 0.0
        return 0, 0

    dev_a = angle - detector._neutral_angle
    dev_l = (hlen - detector._neutral_hlen) / detector._neutral_hlen

    dz = s.get("dead_zone", 0.03)
    if abs(dev_a) < dz:
        dev_a = 0.0
    if abs(dev_l) < dz:
        dev_l = 0.0

    if detector._sx is None:
        detector._sx = 0.0
        detector._sy = 0.0
    sm = s.get("smooth", 0.3)
    detector._sx += sm * (dev_a * s.get("sensitivity_x", 2.0) - detector._sx)
    detector._sy += sm * (dev_l * s.get("sensitivity_y", 1.5) - detector._sy)

    return max(-127, min(127, int(detector._sx * 127))), max(-127, min(127, int(detector._sy * 127)))


def _mode_tilt_vector(detector, frame, lm, s):
    """Вектор запястье→центр ладони как стик."""
    wx, wy = lm[0].x, lm[0].y
    px = (lm[5].x + lm[9].x + lm[13].x + lm[17].x) / 4
    py = (lm[5].y + lm[9].y + lm[13].y + lm[17].y) / 4

    vx = px - wx
    vy = py - wy
    length = math.hypot(vx, vy)
    if length < 0.01:
        return 0, 0

    nx, ny = vx / length, vy / length

    # сглаживание
    if detector._sx is None:
        detector._sx = 0.0
        detector._sy = 0.0
    sm = s.get("smooth", 0.3)
    detector._sx += sm * (nx - detector._sx)
    detector._sy += sm * (ny - detector._sy)
    snx, sny = detector._sx, detector._sy

    # угол от вертикали
    angle = math.atan2(snx, -sny)
    dz = s.get("dead_zone", 0.03)
    if abs(angle) < dz:
        return 0, 0

    tilt = min(1.0, abs(angle) / (math.pi / 2))
    speed = tilt * 127.0 * s.get("sensitivity_x", 2.0)
    speed = min(127.0, speed)

    dx = int(math.sin(angle) * speed)
    dy = int(-math.cos(angle) * speed)
    return dx, dy


def _mode_delta(detector, frame, lm, s):
    """Магнит: движение руки между кадрами → курсор."""
    h, w, _ = frame.shape
    cx = lm[9].x * w
    cy = lm[9].y * h

    sm = s.get("smooth", 0.3)
    if detector._sx is None:
        detector._sx = cx
        detector._sy = cy
        return 0, 0

    prev_x, prev_y = detector._sx, detector._sy
    detector._sx += sm * (cx - detector._sx)
    detector._sy += sm * (cy - detector._sy)

    dx = (detector._sx - prev_x) * 0.8
    dy = (detector._sy - prev_y) * 0.8

    dz = 2.0  # мёртвая зона в пикселях для дельты
    if abs(dx) < dz:
        dx = 0.0
    if abs(dy) < dz:
        dy = 0.0

    # краевой подталкиватель
    nx, ny = cx / w, cy / h
    eb = s.get("edge_boost", 10.0)
    edge_t = 0.20
    if nx < edge_t:
        dx -= eb * (1 - nx / edge_t) ** 2
    elif nx > 1 - edge_t:
        dx += eb * ((nx - (1 - edge_t)) / edge_t) ** 2
    if ny < edge_t:
        dy -= eb * (1 - ny / edge_t) ** 2
    elif ny > 1 - edge_t:
        dy += eb * ((ny - (1 - edge_t)) / edge_t) ** 2

    sens = s.get("sensitivity_x", 2.0)
    return max(-127, min(127, int(dx * sens))), max(-127, min(127, int(dy * sens)))


def _mode_position(detector, frame, lm, s):
    """Позиция: смещение руки от центра кадра → скорость курсора."""
    h, w, _ = frame.shape
    cx, cy = w // 2, h // 2
    hx = lm[9].x * w
    hy = lm[9].y * h

    max_range = 150
    x = max(-127, min(127, int(((hx - cx) / max_range) * 127)))
    y = max(-127, min(127, int(((hy - cy) / max_range) * 127)))

    dz_px = int(s.get("dead_zone", 0.03) * 127)
    if abs(x) < dz_px:
        x = 0
    if abs(y) < dz_px:
        y = 0

    # краевой подталкиватель
    nx, ny = hx / w, hy / w
    eb = s.get("edge_boost", 10.0)
    edge_t = 0.20
    if nx < edge_t:
        x -= int(eb * (1 - nx / edge_t) ** 2)
    elif nx > 1 - edge_t:
        x += int(eb * ((nx - (1 - edge_t)) / edge_t) ** 2)
    if ny < edge_t:
        y -= int(eb * (1 - ny / edge_t) ** 2)
    elif ny > 1 - edge_t:
        y += int(eb * ((ny - (1 - edge_t)) / edge_t) ** 2)

    sens = s.get("sensitivity_x", 2.0)
    return max(-127, min(127, int(x * sens))), max(-127, min(127, int(y * sens)))


# ═══════════════════════════════════════════════════════════════════════════════
#  HandDetector
# ═══════════════════════════════════════════════════════════════════════════════

MODES = {
    "tilt": _mode_tilt,
    "tilt-vector": _mode_tilt_vector,
    "delta": _mode_delta,
    "position": _mode_position,
}


class HandDetector:
    """Поддержка 4 режимов управления. Настройки из settings.json."""

    def __init__(self):
        s = _load_settings()
        conf = s.get("confidence", 0.7)
        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=MODEL_PATH),
            num_hands=1,
            min_hand_detection_confidence=conf,
            min_hand_presence_confidence=conf,
            min_tracking_confidence=conf,
        )
        self.model = HandLandmarker.create_from_options(options)

        # общее состояние
        self._calibrated = False
        self._neutral_angle: float = 0.0
        self._neutral_hlen: float = 0.0
        self._sx: float | None = None
        self._sy: float | None = None
        self._current_mode: str | None = None

    def detect(self, frame):
        """(x, y, button) — дельта курсора -127..127 и флаг клика."""
        s = _load_settings()
        mode = s.get("mode", "tilt")

        # ── сброс состояния при смене режима ──
        if mode != self._current_mode:
            self._current_mode = mode
            self._calibrated = False
            self._neutral_angle = 0.0
            self._neutral_hlen = 0.0
            self._sx = None
            self._sy = None

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self.model.detect(mp_img)

        if not result.hand_landmarks:
            self._calibrated = False
            self._sx = None
            self._sy = None
            return 0, 0, False

        lm = result.hand_landmarks[0]

        # выбираем режим
        mode_fn = MODES.get(mode, _mode_tilt)
        x, y = mode_fn(self, frame, lm, s)

        # пинч
        d = math.hypot(lm[4].x - lm[8].x, lm[4].y - lm[8].y)
        button = d < s.get("pinch_threshold", 0.05)

        return x, y, button
