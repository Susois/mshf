"""Suy NHÃN DÒNG bị tấn công (ground-truth line-level) một cách tất định.

Tấn công trong VEDTD là tất định, nên có thể suy nhãn dòng "vàng" (thật sự bị
chỉnh sửa) bằng cách so text SẠCH của bản gốc với text SẠCH của candidate
(2.ground_truth), độc lập với nhiễu OCR. Nhãn này chỉ dùng để ĐÁNH GIÁ localization
(line-level precision/recall/F1), KHÔNG dùng làm input suy luận.

Quy ước trả về: tập chỉ số dòng candidate (0-indexed theo read_text_lines của GT
candidate) được coi là bị tác động:
  - insert : dòng chỉ có ở candidate  -> "inserted"
  - delete : dòng chỉ có ở original   -> "deleted" (đánh dấu tại vị trí ghép trên candidate)
  - modify : cặp dòng ghép nhưng có thay đổi và<->hoặc / phủ định / số / cer cao
"""
from __future__ import annotations

from mshf.core import io_utils
from mshf.core.line_align import match_lines
from mshf.core.semantic_risk import semantic_risk
import config


def ground_truth_spans(doc_id: str, category: str) -> dict:
    """Trả nhãn dòng bị tấn công suy từ GT sạch (original vs candidate).

    Kết quả:
      {
        "candidate_line_count": int,
        "inserted": [cand_idx, ...],   # dòng chèn thêm (chỉ có ở candidate)
        "modified": [cand_idx, ...],   # dòng bị sửa nội dung
        "deleted_after": [cand_idx,],  # vị trí candidate NGAY SAU chỗ có dòng bị xoá
        "positive": set(cand_idx),     # hợp của inserted+modified (dòng "vàng" trên candidate)
      }
    """
    ref_lines = io_utils.read_text_lines(io_utils.gt_text_path(config.ORIGINAL_CAT, doc_id))
    cand_lines = io_utils.read_text_lines(io_utils.gt_text_path(category, doc_id))

    # Gán chỉ số candidate cho từng dòng để traceback ra vị trí.
    cand_dicts = [{"text": t, "bbox": [0.0, float(i), 1.0, float(i) + 1.0], "idx": i}
                  for i, t in enumerate(cand_lines)]
    ref_dicts = io_utils.as_line_dicts(ref_lines)

    pairs = match_lines(ref_dicts, cand_dicts, sort_by_y=False)

    inserted: list[int] = []
    modified: list[int] = []
    deleted_after: list[int] = []

    for p in pairs:
        if p["type"] == "inserted":
            inserted.append(p["cand_line"]["idx"])
        elif p["type"] == "deleted":
            # dòng bị xoá không còn trên candidate; đánh dấu điểm neo là dòng
            # candidate kế tiếp (nếu có) để đo recall xoá.
            deleted_after.append(_next_cand_idx(pairs, p))
        else:  # match
            ref_txt = p["orig_line"]["text"]
            cand_txt = p["cand_line"]["text"]
            risk = semantic_risk(ref_txt, cand_txt)
            # GT sạch nên cer thực sự phản ánh thay đổi nội dung (không phải nhiễu OCR).
            from mshf.core.line_align import line_cer
            if risk["flags"] or line_cer(ref_txt, cand_txt) > 1e-6:
                modified.append(p["cand_line"]["idx"])

    positive = set(inserted) | set(modified)
    return {
        "candidate_line_count": len(cand_lines),
        "inserted": inserted,
        "modified": modified,
        "deleted_after": [d for d in deleted_after if d is not None],
        "positive": positive,
    }


def _next_cand_idx(pairs: list[dict], deleted_pair: dict) -> int | None:
    """Tìm chỉ số candidate của dòng match/insert đầu tiên sau một dòng bị xoá."""
    seen = False
    for p in pairs:
        if p is deleted_pair:
            seen = True
            continue
        if seen and p["cand_line"] is not None:
            return p["cand_line"].get("idx")
    return None
