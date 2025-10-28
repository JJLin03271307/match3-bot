# platform_adapters/win_window.py
import win32api
import win32con
import win32gui


def _enum_windows_by_title(substr: str):
    out = []
    s = (substr or "").lower()

    def _cb(hwnd, _):
        if win32gui.IsWindowVisible(hwnd):
            t = win32gui.GetWindowText(hwnd)
            if s in t.lower():
                out.append((hwnd, t))

    win32gui.EnumWindows(_cb, None)
    return out


def find_window(title_substring: str):
    """返回 (hwnd, title) 或 (None, '')"""
    hits = _enum_windows_by_title(title_substring)
    return hits[0] if hits else (None, "")


def ensure_front(hwnd, keep_topmost=False):
    """恢复 + 抢前台；可选保持置顶"""
    if not hwnd: return False
    win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
    flags = win32con.SWP_NOMOVE | win32con.SWP_NOSIZE | win32con.SWP_SHOWWINDOW
    if keep_topmost:
        win32gui.SetWindowPos(hwnd, win32con.HWND_TOPMOST, 0, 0, 0, 0, flags)
    else:
        # 短暂置顶帮助抢前台
        win32gui.SetWindowPos(hwnd, win32con.HWND_TOPMOST, 0, 0, 0, 0, flags)
    # 发送 ALT 允许前台切换
    win32api.keybd_event(win32con.VK_MENU, 0, 0, 0)
    win32api.keybd_event(win32con.VK_MENU, 0, win32con.KEYEVENTF_KEYUP, 0)
    try:
        win32gui.BringWindowToTop(hwnd)
        win32gui.SetForegroundWindow(hwnd)
        win32gui.SetActiveWindow(hwnd)
    except Exception:
        pass
    if not keep_topmost:
        win32gui.SetWindowPos(hwnd, win32con.HWND_NOTOPMOST, 0, 0, 0, 0, flags)
    return True


def get_client_origin(hwnd):
    """
    返回窗口客户区的屏幕坐标与尺寸: (x0, y0, w, h)
    x0,y0 = ClientToScreen(hwnd,(0,0))；w,h 来自 GetClientRect
    """
    if not hwnd: return 0, 0, 0, 0
    l, t, r, b = win32gui.GetClientRect(hwnd)
    x0, y0 = win32gui.ClientToScreen(hwnd, (0, 0))
    return x0, y0, r - l, b - t
