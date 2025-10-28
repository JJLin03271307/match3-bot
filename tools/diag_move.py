# tools/diag_move.py  —— 以“窗口客户区”为锚点的定位与点击自检
import os, sys, time, yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import win32gui, win32con, win32api
import mss, pyautogui
import numpy as np

from platform_adapters.windows import WinScreen, WinInput
from core.vision import autolocate_board

GAME_TITLE_SUBSTR = "Bejeweled"  # 窗口标题关键字，必要可改成更精确
CFG_PATH = "games/sample_match3.yaml"


# —— Win32 小工具 —— #
def find_window(title_substr: str):
    hits = []
    s = title_substr.lower()

    def cb(hwnd, _):
        if win32gui.IsWindowVisible(hwnd):
            t = win32gui.GetWindowText(hwnd)
            if s in t.lower():
                hits.append((hwnd, t))

    win32gui.EnumWindows(cb, None)
    return hits


def ensure_front(hwnd):
    win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
    win32gui.SetWindowPos(hwnd, win32con.HWND_TOPMOST, 0, 0, 0, 0,
                          win32con.SWP_NOMOVE | win32con.SWP_NOSIZE | win32con.SWP_SHOWWINDOW)
    # 发送 ALT 让系统允许切前台
    win32api.keybd_event(win32con.VK_MENU, 0, 0, 0)
    win32api.keybd_event(win32con.VK_MENU, 0, win32con.KEYEVENTF_KEYUP, 0)
    try:
        win32gui.BringWindowToTop(hwnd)
        win32gui.SetForegroundWindow(hwnd)
        win32gui.SetActiveWindow(hwnd)
    except Exception:
        pass


def get_client_origin(hwnd):
    # 返回 (client_x0, client_y0, client_w, client_h) —— 全屏绝对坐标 + 尺寸
    l, t, r, b = win32gui.GetClientRect(hwnd)
    x0, y0 = win32gui.ClientToScreen(hwnd, (0, 0))
    return x0, y0, r - l, b - t


def pick_monitor_for_point(x, y):
    sct = mss.mss()
    choice = 1
    for i, mon in enumerate(sct.monitors[1:], start=1):
        lx, ty, w, h = mon["left"], mon["top"], mon["width"], mon["height"]
        if lx <= x < lx + w and ty <= y < ty + h:
            choice = i;
            break
    return choice


def main():
    with open(CFG_PATH, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # 1) 找窗口并拉前台
    wins = find_window(GAME_TITLE_SUBSTR)
    if not wins:
        print("[ERR] 找不到窗口。请确认标题关键字：", GAME_TITLE_SUBSTR)
        return
    hwnd, title = wins[0]
    print("[WIN]", hex(hwnd), title)
    ensure_front(hwnd)
    time.sleep(0.15)

    # 2) 取窗口客户区信息与所在显示器
    cx0, cy0, cw, ch = get_client_origin(hwnd)
    mon_idx = pick_monitor_for_point(cx0 + cw // 2, cy0 + ch // 2)
    print(f"[CLIENT] origin=({cx0},{cy0}) size={cw}x{ch}  [MON]={mon_idx}")

    # 3) 建立截图/输入（含 DPI 映射）
    screen = WinScreen(monitor=mon_idx)
    inputc = WinInput(screen.bbox())
    py_w, py_h = pyautogui.size()
    mon = screen.bbox()
    sx = py_w / mon['width'];
    sy = py_h / mon['height']
    print(
        f"[MAP] py={py_w}x{py_h}, mon={mon['width']}x{mon['height']} @ ({mon['left']},{mon['top']}), scale=({sx:.3f},{sy:.3f})")

    # 4) 截全屏 -> 裁客户区图像 -> 在客户区里自动定位
    full = screen.grab()
    xs = cx0 - mon['left'];
    ys = cy0 - mon['top']
    # 边界保护
    xs = max(0, min(xs, full.shape[1] - 1))
    ys = max(0, min(ys, full.shape[0] - 1))
    xe = min(xs + cw, full.shape[1])
    ye = min(ys + ch, full.shape[0])
    client_img = full[ys:ye, xs:xe].copy()

    roi_rel, cell = autolocate_board(client_img, rows=cfg["board"]["rows"], cols=cfg["board"]["cols"])
    rl, rt, rw, rh = roi_rel
    # 相对客户区 -> 全屏绝对坐标
    l = cx0 + rl
    t = cy0 + rt
    w = rw
    h = rh
    print(f"[AutoLocate@client] rel=({rl},{rt},{rw},{rh}) cell={cell}  ->  abs=({l},{t},{w},{h})")

    # 5) 点击四角+中心（用绝对坐标），再拖一格
    pts = [(l + 5, t + 5), (l + w - 5, t + 5), (l + w - 5, t + h - 5), (l + 5, t + h - 5), (l + w // 2, t + h // 2)]
    ensure_front(hwnd)
    for i, (x, y) in enumerate(pts, 1):
        print(f"[Click#{i}] ({x},{y})")
        inputc.click(x, y)
        time.sleep(0.5)

    # 中心向右拖一个 cell
    cw_cell = ch_cell = cell
    x1, y1 = (l + w // 2 - cw_cell // 2, t + h // 2)
    x2, y2 = (x1 + cw_cell, y1)
    ensure_front(hwnd)
    print(f"[Drag] ({x1},{y1}) -> ({x2},{y2})")
    inputc.drag(x1, y1, x2, y2, ms=220)


if __name__ == "__main__":
    main()
