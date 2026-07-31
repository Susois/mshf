"""Căn chỉnh dòng bằng banded Dynamic Programming (port độc lập từ đề tài gốc).

Nguồn logic: Tuan6/detector_and_explainer/explainer.py (match_lines, line_cer).
Port sang MSHF để folder chạy độc lập, không import chéo Tuan6.

Trả về danh sách "pairs", mỗi phần tử là dict:
    {orig_line, cand_line, cer, type}   với type ∈ {match, inserted, deleted}
Dòng "match" có cer > CER_MODIFIED_THRESHOLD được coi là "modified" ở tầng phân loại.
"""
from __future__ import annotations

from mshf.core.io_utils import normalize


def _edit_distance(a: str, b: str) -> int:
    """Levenshtein trên ký tự. Dùng rapidfuzz nếu có (nhanh), ngược lại DP thuần."""
    try:
        from rapidfuzz.distance import Levenshtein

        return Levenshtein.distance(a, b)
    except ImportError:
        pass
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        curr = [i]
        for j, cb in enumerate(b, 1):
            curr.append(min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = curr
    return prev[-1]


def line_cer(ref: str, hyp: str) -> float:
    r, h = normalize(ref), normalize(hyp)
    if not r:
        return 0.0 if not h else 1.0
    return _edit_distance(r, h) / len(r)


def match_lines(orig_lines: list[dict], cand_lines: list[dict], sort_by_y: bool = False) -> list[dict]:
    """Banded DP alignment. orig/cand là list dict có key 'text' (và 'bbox' nếu sort_by_y).

    sort_by_y=True: sắp xếp theo toạ độ y (dùng khi có bbox thật).
    sort_by_y=False: giữ nguyên thứ tự đọc (dùng cho text-only OCR lines).
    """
    if sort_by_y:
        orig_sorted = sorted(orig_lines, key=lambda l: l["bbox"][1])
        cand_sorted = sorted(cand_lines, key=lambda l: l["bbox"][1])
    else:
        orig_sorted = list(orig_lines)
        cand_sorted = list(cand_lines)

    n, m = len(orig_sorted), len(cand_sorted)
    band_width = max(50, abs(n - m) * 3 + 20)
    INF = float("inf")

    dp: dict[tuple[int, int], float] = {(0, 0): 0.0}
    for i in range(1, n + 1):
        if i <= band_width:
            dp[(i, 0)] = float(i)
    for j in range(1, m + 1):
        if j <= band_width:
            dp[(0, j)] = float(j)

    cer_cache: dict[tuple[int, int], float] = {}

    def cached_cer(i_idx: int, j_idx: int) -> float:
        key = (i_idx, j_idx)
        if key not in cer_cache:
            cer_cache[key] = line_cer(orig_sorted[i_idx]["text"], cand_sorted[j_idx]["text"])
        return cer_cache[key]

    for i in range(1, n + 1):
        diag_j = int(i * m / n) if n > 0 else i
        j_lo = max(1, diag_j - band_width)
        j_hi = min(m, diag_j + band_width)
        for j in range(j_lo, j_hi + 1):
            cost = INF
            if (i - 1, j - 1) in dp:
                cost = min(cost, dp[(i - 1, j - 1)] + cached_cer(i - 1, j - 1))
            if (i - 1, j) in dp:
                cost = min(cost, dp[(i - 1, j)] + 1.0)
            if (i, j - 1) in dp:
                cost = min(cost, dp[(i, j - 1)] + 1.0)
            if cost < INF:
                dp[(i, j)] = cost

    pairs: list[dict] = []
    i, j = n, m
    while i > 0 or j > 0:
        matched = False
        if i > 0 and j > 0 and (i - 1, j - 1) in dp and (i, j) in dp:
            cer = cached_cer(i - 1, j - 1)
            if abs(dp[(i, j)] - (dp[(i - 1, j - 1)] + cer)) < 1e-9:
                pairs.append({
                    "orig_line": orig_sorted[i - 1],
                    "cand_line": cand_sorted[j - 1],
                    "cer": cer,
                    "type": "match",
                })
                i -= 1
                j -= 1
                matched = True
        if not matched and i > 0 and (i - 1, j) in dp and (i, j) in dp:
            if abs(dp[(i, j)] - (dp[(i - 1, j)] + 1.0)) < 1e-9:
                pairs.append({"orig_line": orig_sorted[i - 1], "cand_line": None, "cer": 1.0, "type": "deleted"})
                i -= 1
                matched = True
        if not matched and j > 0:
            pairs.append({"orig_line": None, "cand_line": cand_sorted[j - 1], "cer": 1.0, "type": "inserted"})
            j -= 1

    pairs.reverse()
    return pairs
