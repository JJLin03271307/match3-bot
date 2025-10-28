# core/vision.py
from typing import Dict

import cv2
import glob
import numpy as np
import os


# ---------- 切格 ----------
def _slice_grid(board_img: np.ndarray, rows: int, cols: int, cell_w: int, cell_h: int, pad_x=0, pad_y=0):
    cells = []
    for r in range(rows):
        row = []
        for c in range(cols):
            x = c * cell_w + pad_x // 2
            y = r * cell_h + pad_y // 2
            patch = board_img[y:y + cell_h - pad_y, x:x + cell_w - pad_x]
            row.append(patch)
        cells.append(row)
    return cells


# ---------- 模板分类（可选） ----------
class TemplateClassifier:
    def __init__(self, templates_dir: str):
        self.templates = []
        if os.path.isdir(templates_dir):
            for path in glob.glob(os.path.join(templates_dir, "*.png")):
                label = os.path.basename(path).split("_")[0]
                img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    self.templates.append((label, img))

    def classify_patch(self, patch_bgr: np.ndarray) -> str:
        if not self.templates: return "unk"
        gray = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2GRAY)
        best_label, best_score = "unk", -1.0
        for label, tpl in self.templates:
            th, tw = tpl.shape[:2]
            ph, pw = gray.shape[:2]
            if th > ph or tw > pw:
                continue
            res = cv2.matchTemplate(gray, tpl, cv2.TM_CCOEFF_NORMED)
            score = float(res.max())
            if score > best_score:
                best_score, best_label = score, label
        return best_label


def classify_grid_template(board_img, cfg, classifier: TemplateClassifier):
    rows, cols = cfg["board"]["rows"], cfg["board"]["cols"]
    cell_w, cell_h = cfg["board"]["cell_w"], cfg["board"]["cell_h"]
    pad_x, pad_y = cfg["board"].get("pad_x", 0), cfg["board"].get("pad_y", 0)
    patches = _slice_grid(board_img, rows, cols, cell_w, cell_h, pad_x, pad_y)
    ids = np.zeros((rows, cols), dtype=np.int32)
    label_to_id: Dict[str, int] = {}
    next_id = 0
    for r in range(rows):
        for c in range(cols):
            lab = classifier.classify_patch(patches[r][c])
            if lab not in label_to_id:
                label_to_id[lab] = next_id
                next_id += 1
            ids[r, c] = label_to_id[lab]
    return ids, label_to_id


# ---------- 逐帧聚类（推荐） ----------
def _feat(patch_bgr: np.ndarray, margin: int = 6):
    """颜色+形状的稳健特征：HSV色相直方图 + Lab(ab)均值 + 边缘方向直方图。"""
    h, w = patch_bgr.shape[:2]
    y0 = margin
    y1 = max(margin, h - margin)
    x0 = margin
    x1 = max(margin, w - margin)
    roi = patch_bgr[y0:y1, x0:x1]

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    hch, sch, vch = cv2.split(hsv)
    mask = (sch > 40)  # 去低饱和背景
    hist_h = cv2.calcHist([hch], [0], mask.astype(np.uint8), [18], [0, 180]).flatten()
    hist_h = hist_h / (hist_h.sum() + 1e-6)

    lab = cv2.cvtColor(roi, cv2.COLOR_BGR2Lab)
    a = lab[:, :, 1].astype(np.float32).mean()
    b = lab[:, :, 2].astype(np.float32).mean()

    g = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy) + 1e-6
    ang = (np.rad2deg(np.arctan2(gy, gx)) + 180.0) % 180.0
    bins = np.linspace(0, 180, 9)  # 8 bins
    hist_e = np.histogram(ang, bins=bins, weights=mag)[0].astype(np.float32)
    hist_e = hist_e / (hist_e.sum() + 1e-6)

    return np.concatenate([hist_h.astype(np.float32), np.array([a, b], dtype=np.float32), hist_e], axis=0)


def _kmeans_best_k(X: np.ndarray, kmin=6, kmax=8):
    """Davies-Bouldin 指数选 k（越小越好）。"""
    best_db = 1e9
    best_labels = best_centers = None
    best_k = kmin
    for k in range(kmin, kmax + 1):
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.2)
        _, labels, centers = cv2.kmeans(X.astype(np.float32), k, None, criteria, 5, cv2.KMEANS_PP_CENTERS)
        labels = labels.reshape(-1)
        # DB index
        S = np.zeros(k, dtype=np.float32)
        for i in range(k):
            Xi = X[labels == i]
            S[i] = 0.0 if len(Xi) == 0 else np.mean(np.linalg.norm(Xi - centers[i], axis=1))
        M = np.linalg.norm(centers[:, None, :] - centers[None, :, :], axis=2) + 1e-6
        R = (S[:, None] + S[None, :]) / M
        for i in range(k): R[i, i] = -1e9
        DB = np.mean(np.max(R, axis=1))
        if DB < best_db:
            best_db, best_k = DB, k
            best_labels, best_centers = labels.copy(), centers.copy()
    return best_k, best_labels, best_centers


class PerFrameClassifier:
    """每帧独立聚类 -> id 网格（不使用历史），稳定且不漂移。"""

    def __init__(self, kmin=6, kmax=8):
        self.kmin = kmin
        self.kmax = kmax

    def classify_grid(self, board_img, cfg):
        rows, cols = cfg["board"]["rows"], cfg["board"]["cols"]
        cw, ch = cfg["board"]["cell_w"], cfg["board"]["cell_h"]
        pad_x, pad_y = cfg["board"].get("pad_x", 0), cfg["board"].get("pad_y", 0)
        patches = _slice_grid(board_img, rows, cols, cw, ch, pad_x, pad_y)

        feats = []
        for r in range(rows):
            for c in range(cols):
                feats.append(_feat(patches[r][c], margin=max(4, cw // 10)))
        X = np.vstack(feats)
        k, labels, _ = _kmeans_best_k(X, self.kmin, self.kmax)
        ids = labels.reshape(rows, cols).astype(np.int32)
        label_to_id = {f"c{i}": int(i) for i in range(k)}
        return ids, label_to_id


# ---------- 统一入口 ----------
def classify_grid(board_img, cfg, classifier):
    method = cfg.get("tiles", {}).get("method", "perframe").lower()
    if method == "template" and isinstance(classifier, TemplateClassifier):
        return classify_grid_template(board_img, cfg, classifier)
    elif method == "perframe" and isinstance(classifier, PerFrameClassifier):
        return classifier.classify_grid(board_img, cfg)
    else:
        # 兜底：尝试逐帧聚类
        pf = PerFrameClassifier()
        return pf.classify_grid(board_img, cfg)


# ---------- 自动定位（边缘能量法） ----------
def autolocate_board(full_bgr, rows=8, cols=8, cell_min=48, cell_max=96):
    gray = cv2.cvtColor(full_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    sx = cv2.Sobel(blur, cv2.CV_32F, 1, 0, ksize=3)
    sy = cv2.Sobel(blur, cv2.CV_32F, 0, 1, ksize=3)
    col_e = np.abs(sx).sum(axis=0)
    row_e = np.abs(sy).sum(axis=1)

    def _smooth(v, k=51):
        k = k if k % 2 == 1 else k + 1
        return cv2.GaussianBlur(v.astype(np.float32).reshape(1, -1), (k, 1), 0).ravel()

    col_s = _smooth(col_e, 51)
    row_s = _smooth(row_e, 51)
    col_thr = np.percentile(col_s, 92)
    row_thr = np.percentile(row_s, 92)
    col_mask = col_s > col_thr
    row_mask = row_s > row_thr

    def _segments(mask):
        segs = []
        st = None
        for i, m in enumerate(mask):
            if m and st is None:
                st = i
            elif not m and st is not None:
                segs.append((st, i - 1)); st = None
        if st is not None: segs.append((st, len(mask) - 1))
        return segs

    H, W = full_bgr.shape[:2]
    csegs = _segments(col_mask)
    rsegs = _segments(row_mask)

    def _pick(segs, total):
        if not segs: return None
        center = total / 2
        best = None
        best_score = -1
        for a, b in segs:
            width = b - a + 1
            mid = (a + b) / 2
            score = width - abs(mid - center) * 0.25
            if score > best_score: best_score = score; best = (a, b)
        return best

    cseg = _pick(csegs, W) or (int(W * 0.35), int(W * 0.9))
    rseg = _pick(rsegs, H) or (int(H * 0.05), int(H * 0.95))
    left, right = cseg
    top, bottom = rseg
    width = right - left + 1
    height = bottom - top + 1
    side = max(min(width, height), rows * cell_min)
    side = min(side, rows * cell_max, W, H)
    cx = (left + right) // 2
    cy = (top + bottom) // 2
    left = max(0, cx - side // 2)
    top = max(0, cy - side // 2)
    left = min(left, W - side)
    top = min(top, H - side)
    cell = max(cell_min, min(cell_max, side // rows))
    side = cell * rows
    left = int(np.clip(cx - side // 2, 0, W - side))
    top = int(np.clip(cy - side // 2, 0, H - side))
    return (left, top, side, side), int(cell)
