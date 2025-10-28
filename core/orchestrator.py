# core/orchestrator.py
import cv2
import os
import time
from datetime import datetime
from typing import Tuple

import numpy as np

from core.detector import NoopLevelDetector, SimpleLevelDetector
from core.planner import list_legal_swaps_strict
from core.vision import TemplateClassifier, PerFrameClassifier, classify_grid, autolocate_board
from platform_adapters.win_window import find_window, ensure_front, get_client_origin


class Orchestrator:
    def __init__(self, screen, inputc, cfg):
        self.screen = screen
        self.inputc = inputc
        self.cfg = cfg

        os.makedirs("runs/screenshots", exist_ok=True)
        os.makedirs("runs/debug", exist_ok=True)

        # ---- 识别器：逐帧聚类（稳定不漂移）或模板 ----
        tiles_method = cfg.get("tiles", {}).get("method", "perframe").lower()
        if tiles_method == "template":
            tiles_dir = cfg.get("tiles", {}).get("templates_dir", "games/tiles")
            self.classifier = TemplateClassifier(templates_dir=tiles_dir)
        else:
            # PerFrameClassifier: 每帧独立聚类，不使用历史，防止标签漂移
            self.classifier = PerFrameClassifier(kmin=6, kmax=8)

        # ---- 结束检测 ----
        level_cfg = cfg.get("level_end", {})
        method = level_cfg.get("method", "template")
        if method == "none":
            self.detector = NoopLevelDetector()
        else:
            succ = level_cfg.get("success_templates", [])
            fail = level_cfg.get("fail_templates", [])
            thr = level_cfg.get("thr", 0.9)
            kws = level_cfg.get("ocr_keywords", [])
            use_ocr = True if (kws and method.lower() != "template") else False
            self.detector = SimpleLevelDetector(succ, fail, thr=thr,
                                                use_ocr=use_ocr, keywords=kws)

        # ---- 窗口/ROI ----
        win_cfg = cfg.get("window", {})
        self.win_title = win_cfg.get("title_substring", "Bejeweled")
        self.force_focus_on_start = win_cfg.get("force_focus_on_start", True)
        self.keep_topmost = win_cfg.get("keep_topmost", False)
        self.hwnd, title = find_window(self.win_title)
        print(f"[Window] hwnd={hex(self.hwnd) if self.hwnd else None} title='{title}'")

        bd = cfg["board"]
        self.rows, self.cols = bd["rows"], bd["cols"]
        self.cell_w = bd.get("cell_w", 55)
        self.cell_h = bd.get("cell_h", 55)
        self.anchor_mode = bd.get("anchor", "client")  # "client" | "abs"
        self.autolocate = bd.get("autolocate", False)
        self.roi_abs = tuple(bd.get("roi", [0, 0, 0, 0]))
        self.roi_rel = tuple(bd.get("roi_rel_to_client", [0, 0, 0, 0]))

        # ---- 时序/阈值 ----
        tcfg = cfg.get("timing", {})
        self.drag_ms = tcfg.get("drag_ms", 260)
        self.wait_ms = tcfg.get("post_move_wait_ms", 1100)
        self.settle_check_ms = tcfg.get("settle_check_ms", 250)
        self.settle_patience = tcfg.get("settle_patience", 4)
        self.settle_thresh = float(tcfg.get("settle_thresh", 6.0))
        # 执行后“全局像素差”成功阈值（配合本地两格校验，取较低值）
        self.exec_min_delta = float(tcfg.get("exec_min_delta", 3.5))

        # ---- 统计/截图 ----
        self.cap_every = cfg.get("capture", {}).get("every_moves", 0)
        self.moves = 0
        self._fail_ttl = 6.0  # 建议先用 6.0
        self._failed_cache = {}  # 交换对 → 过期时间
        self._failed_cells = {}  # 新增：单格 → 过期时间

    # ---------- ROI / 窗口 ----------
    def _mean_hsv_center(self, img):
        # 取中心 40% 的方窗避免边缘高亮影响
        h, w = img.shape[:2]
        cx1, cy1 = int(w * 0.3), int(h * 0.3)
        cx2, cy2 = int(w * 0.7), int(h * 0.7)
        patch = img[cy1:cy2, cx1:cx2]
        hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
        return hsv.reshape(-1, 3).mean(axis=0)

    def _would_form_match_by_color(self, roi_abs, board_img, r1, c1, r2, c2, tol=21.0):
        """
        颜色一致性复核：以交换后的“落点”为中心，检查水平/垂直三种形态：
        [-2,-1], [-1,+1], [+1,+2]（垂直同理）。任一满足即通过。
        """
        l, t, w, h = roi_abs
        R, C = self.rows, self.cols

        def cell_crop(r, c, margin=None):
            ax1, ay1, ax2, ay2 = self._cell_rect(roi_abs, r, c, max(3, min(self.cell_w, self.cell_h) // 7))
            x1, y1, x2, y2 = ax1 - l, ay1 - t, ax2 - l, ay2 - t
            return board_img[y1:y2, x1:x2]

        def mean_color(r, c):
            crop = cell_crop(r, c)
            return self._mean_hsv_center(crop)

        def dist(u, v):
            return float(np.linalg.norm(u - v))

        def ok3(center_col, p1, p2):
            (rA, cA), (rB, cB) = p1, p2
            if not (0 <= rA < R and 0 <= cA < C and 0 <= rB < R and 0 <= cB < C):
                return False
            return dist(center_col, mean_color(rA, cA)) < tol and dist(center_col, mean_color(rB, cB)) < tol

        posA = (r2, c2)  # A 将要落到 B 的位置
        posB = (r1, c1)  # B 将要落到 A 的位置
        colA = mean_color(r1, c1)  # 用 A 的“源颜色”作为中心色
        colB = mean_color(r2, c2)  # 用 B 的“源颜色”作为中心色

        # 以落点为中心的 3 种水平+3 种垂直组合
        def forms_at(pos, col):
            r, c = pos
            horiz = [
                ((r, c - 2), (r, c - 1)),  # 左左
                ((r, c - 1), (r, c + 1)),  # 左右（中间成三！）
                ((r, c + 1), (r, c + 2)),  # 右右
            ]
            vert = [
                ((r - 2, c), (r - 1, c)),  # 上上
                ((r - 1, c), (r + 1, c)),  # 上下（中间成三！）
                ((r + 1, c), (r + 2, c)),  # 下下
            ]
            return any(ok3(col, p1, p2) for (p1, p2) in horiz + vert)

        return forms_at(posA, colA) or forms_at(posB, colB)

    def _is_hypercube_cell(self, roi_abs, board_img, r, c) -> bool:
        # 更鲁棒的“特殊盒子/超能宝盒”检测：用边缘方差 + 亮度，而不限制饱和度
        l, t, w, h = roi_abs
        ax1, ay1, ax2, ay2 = self._cell_rect(roi_abs, r, c, max(3, min(self.cell_w, self.cell_h) // 7))
        x1, y1, x2, y2 = ax1 - l, ay1 - t, ax2 - l, ay2 - t
        crop = board_img[y1:y2, x1:x2]
        if crop.size == 0:
            return False

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        lap = cv2.Laplacian(gray, cv2.CV_32F)
        edge_var = float(lap.var())  # 边缘/纹理强度
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        val = float(hsv[:, :, 2].mean())  # 亮度

        # 经验阈值：宝盒有明显纹理边框且较亮（不同皮肤下也稳定）
        return (edge_var > 110.0) and (val > 75.0)

    def _build_hypercube_moves(self, ids, roi_abs, board_img):
        """
        找出盘面上的宝盒；对每个宝盒，挑选“相邻颜色中盘面数量最多”的那一格去交换，
        作为候选招加入。返回与 planner 同格式的列表：[(move, score, info), ...]
        """
        R, C = ids.shape
        # 统计每种 id 的全盘数量
        uniq, cnts = np.unique(ids, return_counts=True)
        hist = {int(k): int(v) for k, v in zip(uniq.tolist(), cnts.tolist())}

        # 找到疑似宝盒的位置（常见：该类数量很少 & 视觉上满足高对比/低饱和）
        hypers = []
        for r in range(R):
            for c in range(C):
                try:
                    if hist.get(int(ids[r, c]), 0) <= 3 and self._is_hypercube_cell(roi_abs, board_img, r, c):
                        hypers.append((r, c))
                except Exception:
                    continue

        results = []
        for (r, c) in hypers:
            # 4 邻域
            neighs = [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]
            # 选盘面“出现次数最多”的邻居颜色
            best = None
            best_freq = -1
            for (nr, nc) in neighs:
                if 0 <= nr < R and 0 <= nc < C:
                    nid = int(ids[nr, nc])
                    # 排除另一个宝盒（避免误判把两个宝盒当同类）
                    if hist.get(nid, 0) <= 3 and self._is_hypercube_cell(roi_abs, board_img, nr, nc):
                        results.append(
                            (((r, c), (nr, nc)), 400.0, {"total": 0, "maxlen": 0, "groups": 0, "kind": "hypercube"}))
                        continue
                    if hist.get(nid, 0) > best_freq:
                        best_freq = hist.get(nid, 0)
                        best = ((r, c), (nr, nc))
            if best is not None:
                # 评分给很高（>100），并附带信息 kind='hypercube'
                score = 150.0 + 0.5 * best_freq
                info = {"total": best_freq, "maxlen": 0, "groups": 0, "kind": "hypercube"}
                results.append((best, score, info))
        return results

    def _grid_change_ratio(self, roi_abs, before_img, after_img, thr=8.0, margin=None, ignore=None):
        if margin is None:
            margin = max(3, min(self.cell_w, self.cell_h) // 7)
        ignore = set(ignore or [])
        l, t, w, h = roi_abs
        changed = 0
        total = self.rows * self.cols

        for r in range(self.rows):
            for c in range(self.cols):
                if (r, c) in ignore:
                    continue
                ax1, ay1, ax2, ay2 = self._cell_rect(roi_abs, r, c, margin)
                x1, y1, x2, y2 = ax1 - l, ay1 - t, ax2 - l, ay2 - t
                b = before_img[y1:y2, x1:x2]
                a = after_img[y1:y2, x1:x2]
                if b.size == 0 or a.size == 0:
                    continue
                bgs = cv2.resize(cv2.cvtColor(b, cv2.COLOR_BGR2GRAY), (28, 28))
                ags = cv2.resize(cv2.cvtColor(a, cv2.COLOR_BGR2GRAY), (28, 28))
                d = float(cv2.absdiff(ags, bgs).mean())
                if d >= thr:
                    changed += 1

        return changed / max(1, total)

    def _abs_roi_from_mode(self, full_frame):
        """根据 anchor 模式返回屏幕绝对 ROI (l,t,w,h)。需要时仅自动定位一次并锁定。"""
        if self.anchor_mode == "client" and self.hwnd:
            x0, y0, cw, ch = get_client_origin(self.hwnd)
            if self.roi_rel == (0, 0, 0, 0) and self.autolocate:
                mon = self.screen.bbox()
                xs = x0 - mon['left']
                ys = y0 - mon['top']
                client_img = full_frame[ys:ys + ch, xs:xs + cw].copy()
                roi_local, cell = autolocate_board(client_img, rows=self.rows, cols=self.cols)
                self.roi_rel = tuple(roi_local)
                self.cell_w = self.cell_h = int(cell)
                self.autolocate = False
                print(
                    f"[AutoLocate@client] rel={self.roi_rel} cell={cell} -> abs=({x0 + self.roi_rel[0]},{y0 + self.roi_rel[1]},{self.roi_rel[2]},{self.roi_rel[3]})")
            l = x0 + self.roi_rel[0]
            t = y0 + self.roi_rel[1]
            w = self.roi_rel[2]
            h = self.roi_rel[3]
            return l, t, w, h
        else:
            if self.autolocate:
                roi, cell = autolocate_board(full_frame, rows=self.rows, cols=self.cols)
                self.roi_abs = tuple(roi)
                self.cell_w = self.cell_h = int(cell)
                self.autolocate = False
                print(f"[AutoLocate@abs] abs={self.roi_abs} cell={cell}")
            return self.roi_abs

    def _cell_center_abs(self, roi_abs: Tuple[int, int, int, int], r: int, c: int):
        l, t, w, h = roi_abs
        return l + c * self.cell_w + self.cell_w // 2, t + r * self.cell_h + self.cell_h // 2

    # ---------- 稳定等待 ----------
    def _wait_board_stable(self, roi_abs):
        """等待棋盘动画结束：ROI 灰度 64×64 的平均差 < 阈值，连续 N 次。"""
        prev = None
        ok = 0
        t0 = time.time()
        while True:
            frame = self.screen.grab_roi(roi_abs)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            small = cv2.resize(gray, (64, 64))
            if prev is not None:
                mad = float(cv2.absdiff(small, prev).mean())
                if mad < self.settle_thresh:
                    ok += 1
                else:
                    ok = 0
                if ok >= self.settle_patience: break
            prev = small
            if (time.time() - t0) * 1000.0 > max(2000, self.wait_ms * 2): break
            time.sleep(self.settle_check_ms / 1000.0)

    # ---------- 签名 & 调试 ----------
    def _roi_signature(self, roi_abs, sz=32):
        img = self.screen.grab_roi(roi_abs)
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return cv2.resize(g, (sz, sz)).astype(np.float32)

    def _save_debug(self, full_frame, roi_abs, move_xy):
        l, t, w, h = roi_abs
        dbg = full_frame.copy()
        # 画网格
        for r in range(1, self.rows):
            y = t + r * self.cell_h
            cv2.line(dbg, (l, y), (l + w, y), (0, 255, 0), 1)
        for c in range(1, self.cols):
            x = l + c * self.cell_w
            cv2.line(dbg, (x, t), (x, t + h), (0, 255, 0), 1)
        # 标记移动
        (x1, y1), (x2, y2) = move_xy
        cv2.arrowedLine(dbg, (x1, y1), (x2, y2), (0, 0, 255), 2, tipLength=0.2)
        ts = datetime.now().strftime("%H%M%S_%f")
        cv2.imwrite(f"runs/debug/move_{ts}.png", dbg)

    # ---------- 本地两格矩形 ----------
    def _cell_rect(self, roi_abs, r, c, margin=None):
        if margin is None:
            margin = max(3, min(self.cell_w, self.cell_h) // 7)  # 自适应边距
        l, t, w, h = roi_abs
        x0 = l + c * self.cell_w + margin
        y0 = t + r * self.cell_h + margin
        x1 = l + (c + 1) * self.cell_w - margin
        y1 = t + (r + 1) * self.cell_h - margin
        return int(x0), int(y0), int(x1), int(y1)

    def _local_swap_ok(self, roi_abs, r1, c1, r2, c2, before_img, after_img, thr=8.0):
        """
        仅比较这两格在交换前后的像素差（降采样到 28×28）。
        注意：before_img/after_img 是 ROI 图，坐标要减去 roi 左上角。
        """
        l, t, w, h = roi_abs

        # 1) 先拿“绝对”格子框
        ax1, ay1, ax2, ay2 = self._cell_rect(roi_abs, r1, c1)
        ax3, ay3, ax4, ay4 = self._cell_rect(roi_abs, r2, c2)

        # 2) 转成 ROI 内的“相对”坐标
        x1, y1, x2, y2 = ax1 - l, ay1 - t, ax2 - l, ay2 - t
        x3, y3, x4, y4 = ax3 - l, ay3 - t, ax4 - l, ay4 - t

        # 3) 做一次边界裁剪，避免越界为空
        H, W = before_img.shape[:2]

        def _clip(x, y, x2, y2):
            x = max(0, min(W, x))
            x2 = max(0, min(W, x2))
            y = max(0, min(H, y))
            y2 = max(0, min(H, y2))
            # 保证至少 2 像素厚度
            if x2 - x < 2: x2 = min(W, x + 2)
            if y2 - y < 2: y2 = min(H, y + 2)
            return int(x), int(y), int(x2), int(y2)

        x1, y1, x2, y2 = _clip(x1, y1, x2, y2)
        x3, y3, x4, y4 = _clip(x3, y3, x4, y4)

        # 4) 裁两格的前后小块
        b1 = before_img[y1:y2, x1:x2]
        b2 = before_img[y3:y4, x3:x4]
        a1 = after_img[y1:y2, x1:x2]
        a2 = after_img[y3:y4, x3:x4]

        # 安全兜底：任何一块为空就直接返回 False，避免崩溃
        if b1.size == 0 or b2.size == 0 or a1.size == 0 or a2.size == 0:
            print("[Local] empty crop -> fallback False")
            return False

        # 5) 转灰度并降采样后计算像素差
        b1s = cv2.resize(cv2.cvtColor(b1, cv2.COLOR_BGR2GRAY), (28, 28))
        b2s = cv2.resize(cv2.cvtColor(b2, cv2.COLOR_BGR2GRAY), (28, 28))
        a1s = cv2.resize(cv2.cvtColor(a1, cv2.COLOR_BGR2GRAY), (28, 28))
        a2s = cv2.resize(cv2.cvtColor(a2, cv2.COLOR_BGR2GRAY), (28, 28))

        d1 = float(cv2.absdiff(a1s, b1s).mean())
        d2 = float(cv2.absdiff(a2s, b2s).mean())
        print(f"[Local] d1={d1:.2f} d2={d2:.2f}")

        return (d1 >= thr) or (d2 >= thr)

    # ---------- 主循环 ----------
    def run_one_level(self, level_id="L1"):
        if self.force_focus_on_start and self.hwnd:
            ensure_front(self.hwnd, keep_topmost=False)

        while True:
            full = self.screen.grab()
            roi_abs = self._abs_roi_from_mode(full)

            # 结束检测
            try:
                end, status = self.detector.is_level_end(full)
            except Exception as e:
                print("[Detector] error:", e)
                end, status = False, ""
            if end:
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                out = f"runs/screenshots/{level_id}_{status}_{ts}.png"
                cv2.imwrite(out, full)
                print("[LevelEnd]", status, "saved:", out)
                return status

            # 等动画停
            self._wait_board_stable(roi_abs)

            # 逐帧聚类 → id 网格
            board_img = self.screen.grab_roi(roi_abs)
            ids, _ = classify_grid(board_img, self.cfg, self.classifier)

            # 穷举“必消招”，只选能立即≥3 的合法交换
            # 生成完整候选列表并过滤黑名单
            cands = list_legal_swaps_strict(ids)
            cands += self._build_hypercube_moves(ids, roi_abs, board_img)
            now = time.time()
            # 清理过期
            self._failed_cache = {k: exp for k, exp in self._failed_cache.items() if exp > now}

            def _key(m):
                a, b = m
                return tuple(sorted([a, b]))

            # 清理过期
            now = time.time()
            self._failed_cache = {k: exp for k, exp in self._failed_cache.items() if exp > now}
            self._failed_cells = {k: exp for k, exp in self._failed_cells.items() if exp > now}

            def _cell_ok(m):
                (r1, c1), (r2, c2) = m
                return (r1, c1) not in self._failed_cells and (r2, c2) not in self._failed_cells

            cands = [(m, sc, info) for (m, sc, info) in cands
                     if _key(m) not in self._failed_cache and _cell_ok(m)]

            if not cands:
                print("[Strict] no legal moves (after blacklist); retry.")
                time.sleep(0.3)
                continue

            (m, sc, info) = cands[0]
            (r1, c1), (r2, c2) = m
            print(f"[Strict] {(r1, c1)} -> {(r2, c2)}  score={sc:.1f} "
                  f"total={info.get('total')} maxlen={info.get('maxlen')} groups={info.get('groups')} "
                  f"kind={info.get('kind', 'strict')}")

            x1, y1 = self._cell_center_abs(roi_abs, r1, c1)
            x2, y2 = self._cell_center_abs(roi_abs, r2, c2)
            # 预检：三连颜色一致性，不通过就拉黑并跳过，不执行拖拽
            is_power = (info.get("kind") == "hypercube")

            # 仅对普通三消做颜色预检；宝盒招直接放行
            if (not is_power) and (not self._would_form_match_by_color(roi_abs, board_img, r1, c1, r2, c2, tol=22.0)):
                expiry = time.time() + self._fail_ttl
                self._failed_cache[_key(((r1, c1), (r2, c2)))] = expiry
                self._failed_cells[(r1, c1)] = expiry
                self._failed_cells[(r2, c2)] = expiry
                print("[PreCheck] veto by color-consistency; blacklist pair & cells.")
                time.sleep(0.2)
                continue

            # —— 执行（稳）：轻点唤醒焦点 + 0.8格 overshoot 拖拽 ——
            ensure_front(self.hwnd, keep_topmost=False)
            self.inputc.click(x1, y1)
            time.sleep(0.05)

            dx = np.sign(x2 - x1) * int(max(8, 0.8 * self.cell_w))
            dy = np.sign(y2 - y1) * int(max(8, 0.8 * self.cell_h))
            x2o, y2o = x2 + dx, y2 + dy

            before_full = self.screen.grab()  # 保存调试图基底
            before_roi = self.screen.grab_roi(roi_abs)  # 执行前 ROI

            sig_before = self._roi_signature(roi_abs)
            self.inputc.drag(x1, y1, x2o, y2o, ms=self.drag_ms)
            time.sleep(max(0.15, self.drag_ms / 1000.0))
            self._save_debug(before_full, roi_abs, ((x1, y1), (x2, y2)))
            self._wait_board_stable(roi_abs)
            after_roi = self.screen.grab_roi(roi_abs)

            # 全局签名差：稳定前后
            sig_after = self._roi_signature(roi_abs)
            delta = float(np.mean(np.abs(sig_after - sig_before)))
            print(f"[Exec] delta_stable={delta:.2f}")

            # 逐格变化率（像素差），不受聚类标签漂移影响
            grid_ratio_ex = self._grid_change_ratio(
                roi_abs, before_roi, after_roi, thr=8.0, ignore=[(r1, c1), (r2, c2)]
            )
            local_hint = self._local_swap_ok(roi_abs, r1, c1, r2, c2, before_roi, after_roi, thr=8.0)
            print(f"[Verify] grid_change_excl={grid_ratio_ex:.3f} delta_stable={delta:.2f} local={local_hint}")

            # 成功条件：稳定后的全局差 + 被交换两格之外至少有明显变化
            success = (delta >= self.exec_min_delta) and (grid_ratio_ex >= 0.02)

            if not success:
                # 标记失败招，短期不再尝试
                expiry = time.time() + self._fail_ttl
                self._failed_cache[_key(((r1, c1), (r2, c2)))] = expiry
                self._failed_cells[(r1, c1)] = expiry
                self._failed_cells[(r2, c2)] = expiry
                print("[Skip] move not executed (no clear). Blacklist pair & cells.")
                time.sleep(0.4)
                continue
