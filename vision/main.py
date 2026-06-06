import json
import sys

from vision.camera import Camera
from vision.detector import HandDetector


def main():
    cam = Camera()
    detector = HandDetector()

    print("🌟 open-tracker запущен", file=sys.stderr)

    try:
        while True:
            frame = cam.read()
            if frame is None:
                break

            x, y, btn, scroll, hscroll = detector.detect(frame)
            cmd = {
                "x": x,
                "y": y,
                "button": btn,
                "scroll": scroll,
                "hscroll": hscroll,
            }
            print(json.dumps(cmd), flush=True)

    except KeyboardInterrupt:
        pass
    finally:
        cam.release()
        print("👋 завершено", file=sys.stderr)


if __name__ == "__main__":
    main()
