# core/planner.py
from typing import List, Tuple, Dict
import numpy as np

Coord = Tuple[int, int]
Move = Tuple[Coord, Coord]


def _find_matches(board: np.ndarray) -> List[List[Coord]]:
    R, C = board.shape
    groups: List[List[Coord]] = []

    # 横向
    for r in range(R):
        c = 0
        while c < C:
            k = 1
            while c + k < C and board[r, c + k] == board[r, c] and board[r, c] != -1:
                k += 1
            if k >= 3:
                groups.append([(r, cc) for cc in range(c, c + k)])
            c += k

    # 纵向
    for c in range(C):
        r = 0
        while r < R:
            k = 1
            while r + k < R and board[r + k, c] == board[r, c] and board[r, c] != -1:
                k += 1
            if k >= 3:
                groups.append([(rr, c) for rr in range(r, r + k)])
            r += k

    return groups


def _center_bias(R: int, C: int, m: Move) -> float:
    (r1, c1), (r2, c2) = m
    cr, cc = (R - 1) / 2.0, (C - 1) / 2.0
    d1 = np.hypot(r1 - cr, c1 - cc)
    d2 = np.hypot(r2 - cr, c2 - cc)
    dmax = np.hypot(cr, cc) + 1e-6
    return 1.0 - (d1 + d2) / (2 * dmax)


def list_legal_swaps_strict(ids: np.ndarray) -> List[Tuple[Move, float, Dict]]:
    """
    穷举相邻交换；只保留“交换后立刻 ≥3”的合法招。
    评分：5消>4消>多组>总数>中心偏好（边缘轻扣分、交叉给小奖励）。
    """
    R, C = ids.shape
    results: List[Tuple[Move, float, Dict]] = []

    for r in range(R):
        for c in range(C):
            for dr, dc in ((0, 1), (1, 0)):
                nr, nc = r + dr, c + dc
                if nr >= R or nc >= C:
                    continue
                if ids[r, c] == ids[nr, nc]:
                    continue  # 同色互换大多无效，先略过以提速

                tmp = ids.copy()
                tmp[r, c], tmp[nr, nc] = tmp[nr, nc], tmp[r, c]
                groups = _find_matches(tmp)
                if not groups:
                    continue

                total = sum(len(g) for g in groups)
                maxlen = max(len(g) for g in groups)
                num_groups = len(groups)

                # 是否有交叉（T/L形）
                inter_bonus = 0.0
                if num_groups >= 2:
                    s0 = set(groups[0])
                    for g in groups[1:]:
                        if s0.intersection(g):
                            inter_bonus = 2.0
                            break

                score = 0.0
                if maxlen >= 5:
                    score += 100.0
                elif maxlen == 4:
                    score += 20.0
                score += 5.0 * (num_groups - 1)
                score += 1.0 * total
                score += 0.8 * _center_bias(R, C, ((r, c), (nr, nc)))
                edge_pen = 0.0
                for rr, cc in [(r, c), (nr, nc)]:
                    if rr in (0, R - 1): edge_pen += 0.15
                    if cc in (0, C - 1): edge_pen += 0.15
                score -= edge_pen
                score += inter_bonus

                info = {"total": total, "maxlen": maxlen, "groups": num_groups}
                results.append((((r, c), (nr, nc)), float(score), info))

    results.sort(key=lambda x: x[1], reverse=True)
    return results


def choose_move_strict(ids: np.ndarray):
    cand = list_legal_swaps_strict(ids)
    if not cand:
        return None, None, {"reason": "no_legal"}
    (m, sc, info) = cand[0]
    info = {"score": sc, **info}
    return m[0], m[1], info
