import math

import cv2
import mediapipe as mp
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions
from mediapipe.tasks.python.core.base_options import BaseOptions

MODEL_PATH = "hand_landmarker.task"
PINCH_THRESHOLD = 0.05

# ── Tilt-joystick ─────────────────────────────────────────────────────────────
# Вектор от запястья к центру ладони задаёт направление и скорость курсора.
# Рука вертикально (пальцы вверх) = нейтраль → курсор стоит.
# Наклон руки = курсор едет в ту же сторону, скорость от угла наклона.
TILT_SENSITIVITY = 3.0      # множитель скорости
TILT_DEAD_ZONE = 0.04       # минимальный угол наклона (от вертикали), рад
TILT_SMOOTH = 0.25          # сглаживание вектора (0..1)


class HandDetector:
    """MediaPipe HandLandmarker. Управление наклоном кисти — как стик."""

    def __init__(self):
        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=MODEL_PATH),
            num_hands=1,
            min_hand_detection_confidence=0.7,
            min_hand_presence_confidence=0.7,
            min_tracking_confidence=0.7,
        )
        self.model = HandLandmarker.create_from_options(options)

        # сглаженный вектор наклона (wrist → palm-center)
        self._svx: float = 0.0
        self._svy: float = 0.0

    # ── helpers ────────────────────────────────────────────────────────────────

    @staticmethod
    def _palm_center(lm) -> tuple[float, float]:
        """Центр ладони как среднее четырёх knuckle-landmarks."""
        cx = (lm[5].x + lm[9].x + lm[13].x + lm[17].x) / 4
        cy = (lm[5].y + lm[9].y + lm[13].y + lm[17].y) / 4
        return cx, cy

    # ── detect ─────────────────────────────────────────────────────────────────

    def detect(self, frame):
        """Принимает BGR-кадр, возвращает (x, y, button).

        x, y — дельта для курсора (в пикселях мыши), -127..127.
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self.model.detect(mp_img)

        if not result.hand_landmarks:
            self._svx = 0.0
            self._svy = 0.0
            return 0, 0, False

        lm = result.hand_landmarks[0]

        # ── tilt-вектор: запястье → центр ладони ──
        wx, wy = lm[0].x, lm[0].y            # запястье
        px, py = self._palm_center(lm)        # центр ладони
        vx = px - wx
        vy = py - wy  # image coords: y↑ вниз

        # длина вектора (нормализующая информация о расстоянии до камеры)
        length = math.hypot(vx, vy)
        if length < 0.01:
            self._svx = 0.0
            self._svy = 0.0
            return 0, 0, False

        nx = vx / length   # 0..1
        ny = vy / length

        # ── сглаживание ──
        self._svx += TILT_SMOOTH * (nx - self._svx)
        self._svy += TILT_SMOOTH * (ny - self._svy)

        snx = self._svx
        sny = self._svy

        # ── угол от вертикали ──
        # вертикаль в image coords: (0, -1) — пальцы вверх
        # atan2(snx, -sny): 0 = вверх, >0 = наклон вправо, <0 = влево
        angle = math.atan2(snx, -sny)  # -π .. π

        # ── величина наклона ──
        if abs(angle) < TILT_DEAD_ZONE:
            return 0, 0, False

        tilt = min(1.0, abs(angle) / (math.pi / 2))  # 0..1, 1 = горизонтально
        speed = tilt * 127.0 * TILT_SENSITIVITY
        speed = min(127.0, speed)

        # ── курсор: направление = направление наклона ──
        dx = int(math.sin(angle) * speed)
        dy = int(-math.cos(angle) * speed)

        # ── пинч ──
        d = math.hypot(lm[4].x - lm[8].x, lm[4].y - lm[8].y)
        button = d < PINCH_THRESHOLD

        return dx, dy, button
