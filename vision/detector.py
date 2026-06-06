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
    "mode": "delta",
    "sensitivity_x": 1.5,
    "sensitivity_y": 1.0,
    "smooth": 0.2,
    "dead_zone": 0.03,
    "confidence": 0.7,
    "edge_boost": 0.0,
    "pinch_threshold": 0.05,
}

# ── жесты ─────────────────────────────────────────────────────────────────────
PALM_FIST_THRESHOLD = 0.08       # fingertip→MCP дистанция для кулака (norm.)
FIST_SCROLL_SENS = 300.0    # множитель движения кулака → скролл
ZOOM_SENSITIVITY = 300.0    # множитель расстояния между руками → зум


def _load_settings() -> dict:
    try:
        with open(SETTINGS_PATH) as f:
            return {**DEFAULT_SETTINGS, **json.load(f)}
    except (FileNotFoundError, json.JSONDecodeError):
        return dict(DEFAULT_SETTINGS)


# ═══════════════════════════════════════════════════════════════════════════════
#  Режимы управления (только курсор)
# ═══════════════════════════════════════════════════════════════════════════════

def _mode_tilt(detector, frame, lm, s):
    h, w, _ = frame.shape
    angle = math.atan2(lm[17].y - lm[5].y, lm[17].x - lm[5].x)
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
    if abs(dev_a) < dz: dev_a = 0.0
    if abs(dev_l) < dz: dev_l = 0.0
    if detector._sx is None:
        detector._sx = 0.0
        detector._sy = 0.0
    sm = s.get("smooth", 0.3)
    detector._sx += sm * (dev_a * s.get("sensitivity_x", 2.0) - detector._sx)
    detector._sy += sm * (dev_l * s.get("sensitivity_y", 1.5) - detector._sy)
    return max(-127, min(127, int(detector._sx * 127))), max(-127, min(127, int(detector._sy * 127)))


def _mode_tilt_vector(detector, frame, lm, s):
    wx, wy = lm[0].x, lm[0].y
    px = (lm[5].x + lm[9].x + lm[13].x + lm[17].x) / 4
    py = (lm[5].y + lm[9].y + lm[13].y + lm[17].y) / 4
    vx = px - wx; vy = py - wy
    length = math.hypot(vx, vy)
    if length < 0.01: return 0, 0
    nx, ny = vx / length, vy / length
    if detector._sx is None: detector._sx = 0.0; detector._sy = 0.0
    sm = s.get("smooth", 0.3)
    detector._sx += sm * (nx - detector._sx)
    detector._sy += sm * (ny - detector._sy)
    snx, sny = detector._sx, detector._sy
    angle = math.atan2(snx, -sny)
    dz = s.get("dead_zone", 0.03)
    if abs(angle) < dz: return 0, 0
    tilt = min(1.0, abs(angle) / (math.pi / 2))
    speed = tilt * 127.0 * s.get("sensitivity_x", 2.0)
    speed = min(127.0, speed)
    return int(math.sin(angle) * speed), int(-math.cos(angle) * speed)


def _mode_delta(detector, frame, lm, s):
    h, w, _ = frame.shape
    cx = lm[9].x * w; cy = lm[9].y * h
    sm = s.get("smooth", 0.2)
    if detector._sx is None:
        detector._sx = cx; detector._sy = cy
        return 0, 0
    prev_x, prev_y = detector._sx, detector._sy
    detector._sx += sm * (cx - detector._sx)
    detector._sy += sm * (cy - detector._sy)
    dx = (detector._sx - prev_x) * 0.8
    dy = (detector._sy - prev_y) * 0.8
    if abs(dx) < 2.0: dx = 0.0
    if abs(dy) < 2.0: dy = 0.0
    nx, ny = cx / w, cy / h
    eb = s.get("edge_boost", 0.0); edge_t = 0.20
    if nx < edge_t: dx -= eb * (1 - nx / edge_t) ** 2
    elif nx > 1 - edge_t: dx += eb * ((nx - (1 - edge_t)) / edge_t) ** 2
    if ny < edge_t: dy -= eb * (1 - ny / edge_t) ** 2
    elif ny > 1 - edge_t: dy += eb * ((ny - (1 - edge_t)) / edge_t) ** 2
    sens = s.get("sensitivity_x", 1.5)
    return max(-127, min(127, int(dx * sens))), max(-127, min(127, int(dy * sens)))


def _mode_position(detector, frame, lm, s):
    h, w, _ = frame.shape
    cx, cy = w // 2, h // 2
    hx = lm[9].x * w; hy = lm[9].y * h
    max_range = 150
    x = max(-127, min(127, int(((hx - cx) / max_range) * 127)))
    y = max(-127, min(127, int(((hy - cy) / max_range) * 127)))
    dz_px = int(s.get("dead_zone", 0.03) * 127)
    if abs(x) < dz_px: x = 0
    if abs(y) < dz_px: y = 0
    nx, ny = hx / w, hy / w
    eb = s.get("edge_boost", 10.0); edge_t = 0.20
    if nx < edge_t: x -= int(eb * (1 - nx / edge_t) ** 2)
    elif nx > 1 - edge_t: x += int(eb * ((nx - (1 - edge_t)) / edge_t) ** 2)
    if ny < edge_t: y -= int(eb * (1 - ny / edge_t) ** 2)
    elif ny > 1 - edge_t: y += int(eb * ((ny - (1 - edge_t)) / edge_t) ** 2)
    sens = s.get("sensitivity_x", 2.0)
    return max(-127, min(127, int(x * sens))), max(-127, min(127, int(y * sens)))


MODES = {
    "tilt": _mode_tilt,
    "tilt-vector": _mode_tilt_vector,
    "delta": _mode_delta,
    "position": _mode_position,
}


# ═══════════════════════════════════════════════════════════════════════════════
#  HandDetector
# ═══════════════════════════════════════════════════════════════════════════════

class HandDetector:
    """Жесты:
       - открытая ладонь → курсор
       - кулак → скролл
       - щипок thumb+index → клик
       - две руки → зум
    """

    def __init__(self):
        s = _load_settings()
        conf = s.get("confidence", 0.7)
        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=MODEL_PATH),
            num_hands=2,
            min_hand_detection_confidence=conf,
            min_hand_presence_confidence=conf,
            min_tracking_confidence=conf,
        )
        self.model = HandLandmarker.create_from_options(options)

        self._current_mode: str | None = None
        self._calibrated = False
        self._neutral_angle: float = 0.0
        self._neutral_hlen: float = 0.0
        self._sx: float | None = None
        self._sy: float | None = None

        # кулак → скролл
        self._fist_scrolling = False
        self._fist_px = 0.0
        self._fist_py = 0.0

        # зум
        self._zoom_dist: float | None = None

    @staticmethod
    def _palm_center(lm):
        cx = (lm[5].x + lm[9].x + lm[13].x + lm[17].x) / 4
        cy = (lm[5].y + lm[9].y + lm[13].y + lm[17].y) / 4
        return cx, cy

    @staticmethod
    def _is_fist(lm) -> bool:
        """Кулак: 4+ кончиков пальцев близки к центру ладони."""
        cx = (lm[5].x + lm[9].x + lm[13].x + lm[17].x) / 4
        cy = (lm[5].y + lm[9].y + lm[13].y + lm[17].y) / 4
        curled = sum(
            1 for tip_idx in [4, 8, 12, 16, 20]
            if math.hypot(lm[tip_idx].x - cx, lm[tip_idx].y - cy) < PALM_FIST_THRESHOLD
        )
        return curled >= 4

    # ── detect ───────────────────────────────────────────────────────────────

    def detect(self, frame):
        """(x, y, button, scroll, hscroll, ctrl) — всё -127..127/булево."""
        s = _load_settings()
        mode = s.get("mode", "delta")

        if mode != self._current_mode:
            self._current_mode = mode
            self._calibrated = False
            self._neutral_angle = 0.0
            self._neutral_hlen = 0.0
            self._sx = None; self._sy = None

        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self.model.detect(mp_img)

        if not result.hand_landmarks:
            self._calibrated = False
            self._sx = None
            self._fist_scrolling = False
            self._zoom_dist = None
            return 0, 0, False, 0, 0, False

        # ── две руки → зум ──────────────────────────────────────────────────
        if len(result.hand_landmarks) >= 2:
            lm0, lm1 = result.hand_landmarks[0], result.hand_landmarks[1]
            c0x, c0y = self._palm_center(lm0)
            c1x, c1y = self._palm_center(lm1)
            dist = math.hypot(c1x - c0x, c1y - c0y)

            scroll_z = 0
            if self._zoom_dist is not None:
                delta = (dist - self._zoom_dist) * ZOOM_SENSITIVITY
                if abs(delta) >= 1:
                    scroll_z = max(-20, min(20, int(delta)))
            self._zoom_dist = dist
            self._fist_scrolling = False
            return 0, 0, False, scroll_z, 0, True  # ctrl = True = зум

        # ── одна рука ───────────────────────────────────────────────────────
        self._zoom_dist = None
        lm = result.hand_landmarks[0]

        # клик: щипок thumb+index
        click_d = math.hypot(lm[4].x - lm[8].x, lm[4].y - lm[8].y)
        button = click_d < s.get("pinch_threshold", 0.05)

        # ── кулак → скролл ──────────────────────────────────────────────────
        if self._is_fist(lm):
            cx = (lm[5].x + lm[9].x + lm[13].x + lm[17].x) / 4
            cy = (lm[5].y + lm[9].y + lm[13].y + lm[17].y) / 4

            if self._fist_scrolling:
                dx = (cx - self._fist_px) * FIST_SCROLL_SENS
                dy = (cy - self._fist_py) * FIST_SCROLL_SENS
                hsl = max(-20, min(20, int(dx)))
                sl = max(-20, min(20, int(dy)))
                if abs(hsl) < 1: hsl = 0
                if abs(sl) < 1: sl = 0
                if hsl and sl:
                    if abs(hsl) > abs(sl): sl = 0
                    else: hsl = 0
            else:
                self._fist_scrolling = True
                hsl = 0; sl = 0

            self._fist_px = cx; self._fist_py = cy
            return 0, 0, False, sl, hsl, False

        self._fist_scrolling = False

        # ── курсор ──────────────────────────────────────────────────────────
        mode_fn = MODES.get(mode, _mode_delta)
        x, y = mode_fn(self, frame, lm, s)
        return x, y, button, 0, 0, False
