"""Đặc trưng mức DÒNG: so OCR text của candidate với OCR text của bản gốc.

Nhánh B trong MSHF. Dùng chung pipeline OCR cho cả reference và candidate (đúng
khuyến nghị: không lấy ground-truth text làm input suy luận), rồi align dòng để
đếm trực tiếp insert/delete/modify — thứ mà đặc trưng doc-level không định lượng được.
"""
from __future__ import annotations

from mshf.core import io_utils
from mshf.core.line_align import match_lines
from mshf.core.semantic_risk import semantic_risk
import config

# Prefix để tránh trùng tên cột với nhánh gốc / hình học.
LINE_FEATURE_COLS = [
    "ln_insert_count", "ln_delete_count", "ln_modified_count",
    "ln_match_count", "ln_total_ops",
    "ln_insert_ratio", "ln_delete_ratio", "ln_modified_ratio", "ln_unchanged_ratio",
    "ln_mod_cer_mean", "ln_mod_cer_min", "ln_mod_cer_max",
    "ln_negation_count", "ln_conjunction_count", "ln_numeric_count", "ln_critical_count",
]


def _zeros() -> dict:
    return {c: 0.0 for c in LINE_FEATURE_COLS}


def line_features(doc_id: str, category: str) -> dict:
    """Sinh đặc trưng mức dòng cho một cặp (1.original, category) của cùng doc_id."""
    ref_lines = io_utils.read_text_lines(io_utils.ocr_text_path(config.ORIGINAL_CAT, doc_id))
    cand_lines = io_utils.read_text_lines(io_utils.ocr_text_path(category, doc_id))
    return line_features_from_texts(ref_lines, cand_lines)


def line_features_from_texts(ref_lines: list[str], cand_lines: list[str]) -> dict:
    feat = _zeros()
    if not ref_lines and not cand_lines:
        return feat

    pairs = match_lines(
        io_utils.as_line_dicts(ref_lines),
        io_utils.as_line_dicts(cand_lines),
        sort_by_y=False,
    )

    insert = delete = modified = match = 0
    mod_cers: list[float] = []
    neg = conj = numeric = critical = 0

    for p in pairs:
        if p["type"] == "inserted":
            insert += 1
        elif p["type"] == "deleted":
            delete += 1
        else:  # match
            cer = p["cer"]
            ref_txt = p["orig_line"]["text"]
            cand_txt = p["cand_line"]["text"]
            risk = semantic_risk(ref_txt, cand_txt)
            is_modified = cer > config.CER_MODIFIED_THRESHOLD or bool(risk["flags"])
            if is_modified:
                modified += 1
                mod_cers.append(cer)
                flags = risk["flags"]
                if "negation" in flags:
                    neg += 1
                if any(f.startswith("conjunction") for f in flags):
                    conj += 1
                if "numeric_or_date_change" in flags:
                    numeric += 1
                if flags:
                    critical += 1
            else:
                match += 1

    total_lines = insert + delete + modified + match
    denom = max(total_lines, 1)
    feat.update({
        "ln_insert_count": insert,
        "ln_delete_count": delete,
        "ln_modified_count": modified,
        "ln_match_count": match,
        "ln_total_ops": insert + delete + modified,
        "ln_insert_ratio": round(insert / denom, 4),
        "ln_delete_ratio": round(delete / denom, 4),
        "ln_modified_ratio": round(modified / denom, 4),
        "ln_unchanged_ratio": round(match / denom, 4),
        "ln_mod_cer_mean": round(sum(mod_cers) / len(mod_cers), 4) if mod_cers else 0.0,
        "ln_mod_cer_min": round(min(mod_cers), 4) if mod_cers else 0.0,
        "ln_mod_cer_max": round(max(mod_cers), 4) if mod_cers else 0.0,
        "ln_negation_count": neg,
        "ln_conjunction_count": conj,
        "ln_numeric_count": numeric,
        "ln_critical_count": critical,
    })
    return feat
