import cv2


class Camera:
    """Захват кадра с веб-камеры."""

    def __init__(self, device: int = 0):
        self.cap = cv2.VideoCapture(device)

    def read(self):
        ret, frame = self.cap.read()
        if not ret:
            return None
        return cv2.flip(frame, 1)  # зеркало

    def release(self):
        self.cap.release()
