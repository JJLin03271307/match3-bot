# platform_adapters/android.py
import time, subprocess, numpy as np, cv2
from core.interfaces import ScreenProvider, InputController


class AdbScreen(ScreenProvider):
    def __init__(self, serial=None):
        self.serial = serial

    def _adb(self, args):
        base = ["adb"] + (["-s", self.serial] if self.serial else [])
        return subprocess.check_output(base + args)

    def grab(self) -> np.ndarray:
        raw = self._adb(["exec-out", "screencap", "-p"])
        img = np.frombuffer(raw, dtype=np.uint8)
        return cv2.imdecode(img, cv2.IMREAD_COLOR)  # BGR

    def grab_roi(self, roi):
        frame = self.grab()
        l, t, w, h = roi
        return frame[t:t + h, l:l + w, :]


class AdbInput(InputController):
    def __init__(self, serial=None):
        self.serial = serial

    def _adb(self, args):
        base = ["adb"] + (["-s", self.serial] if self.serial else [])
        subprocess.check_call(base + args)

    def click(self, x, y):
        self._adb(["shell", "input", "tap", str(x), str(y)])

    def drag(self, x1, y1, x2, y2, ms=150):
        self._adb(["shell", "input", "swipe", str(x1), str(y1), str(x2), str(y2), str(ms)])
