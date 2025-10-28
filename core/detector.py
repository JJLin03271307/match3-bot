import os, cv2


def _read_gray(p):
    if not p or not os.path.exists(p):
        return None
    im = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
    return im


def _has_template(frame_bgr, tpl, thr=0.90):
    if tpl is None:
        return False
    f = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    fh, fw = f.shape[:2];
    th, tw = tpl.shape[:2]
    if th > fh or tw > fw:
        return False
    res = cv2.matchTemplate(f, tpl, cv2.TM_CCOEFF_NORMED)
    return float(res.max()) >= thr


class NoopLevelDetector:
    def is_level_end(self, frame_bgr):
        return False, ""


class SimpleLevelDetector:
    """Template + (optional) OCR fallback."""

    def __init__(self, success_paths=None, fail_paths=None, thr=0.9, use_ocr=True, keywords=None):
        self.success = [_read_gray(p) for p in (success_paths or [])]
        self.fail = [_read_gray(p) for p in (fail_paths or [])]
        self.thr = thr
        self.use_ocr = use_ocr
        self.keywords = keywords or []
        self.ocr = None
        if use_ocr:
            try:
                import pytesseract
                self.ocr = pytesseract
            except Exception:
                self.use_ocr = False

    def is_level_end(self, frame_bgr):
        for t in self.success:
            if _has_template(frame_bgr, t, self.thr):
                return True, "success"
        for t in self.fail:
            if _has_template(frame_bgr, t, self.thr):
                return True, "fail"
        if self.use_ocr and self.ocr and self.keywords:
            txt = self.ocr.image_to_string(frame_bgr)
            if any(k.lower() in (txt or "").lower() for k in self.keywords):
                return True, "success"
        return False, ""
