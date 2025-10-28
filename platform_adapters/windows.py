# platform_adapters/windows.py
import ctypes
import numpy as np
import mss, cv2, pyautogui
from core.interfaces import ScreenProvider, InputController

try:
    ctypes.windll.user32.SetProcessDPIAware()
except Exception:
    pass


class WinScreen(ScreenProvider):
    def __init__(self, monitor=1):
        self.sct = mss.mss()
        self.mon = self.sct.monitors[monitor]

    def grab(self) -> np.ndarray:
        img = np.array(self.sct.grab(self.mon))
        return img[:, :, :3][:, :, ::-1]

    def grab_roi(self, roi):
        l, t, w, h = roi
        bbox = {"left": self.mon["left"] + l, "top": self.mon["top"] + t, "width": w, "height": h}
        img = np.array(self.sct.grab(bbox))
        return img[:, :, :3][:, :, ::-1]

    def bbox(self):
        return dict(self.mon)


class WinInput(InputController):
    def __init__(self, monitor_bbox):
        self.py_w, self.py_h = pyautogui.size()
        self.mon = monitor_bbox
        self.sx = self.py_w / self.mon['width']
        self.sy = self.py_h / self.mon['height']
        self.offx = self.mon['left'] * self.sx
        self.offy = self.mon['top'] * self.sy
        pyautogui.FAILSAFE = True
        pyautogui.PAUSE = 0.01

    def _map(self, x, y):
        return int(self.offx + x * self.sx), int(self.offy + y * self.sy)

    def drag(self, x1, y1, x2, y2, ms=150):
        X1, Y1 = self._map(x1, y1);
        X2, Y2 = self._map(x2, y2)
        # overshoot 6px，提升触发率
        dx = np.sign(X2 - X1) * 6
        dy = np.sign(Y2 - Y1) * 6
        pyautogui.moveTo(X1, Y1, duration=0.05)
        pyautogui.mouseDown(button='left')
        pyautogui.moveTo(X2 + dx, Y2 + dy, duration=max(ms, 80) / 1000.0)
        pyautogui.mouseUp(button='left')

    def click(self, x, y):
        X, Y = self._map(x, y)
        pyautogui.click(X, Y)
