"""Phát hiện rủi ro ngữ nghĩa mức DÒNG (minh bạch, dựa luật) cho cặp dòng nghi bị sửa.

Bổ sung cho nhánh doc-level của đề tài gốc: khi hai dòng gần giống hệt nhưng đảo
"và"<->"hoặc", thêm/bớt phủ định, hoặc đổi số/ngày, đó là chỉnh sửa nội dung thật
(không phải nhiễu OCR). Đây chính là loại tấn công 4.modify ("và"->"hoặc") mà đặc
trưng trung bình toàn tài liệu bị pha loãng.
"""
from __future__ import annotations

import re

from .io_utils import normalize

_NUMBER = re.compile(r"\d+(?:[.,/]\d+)*")

# term xuất hiện -> nhãn flag
_TERMS = (
    ("không", "negation"),
    ("và", "conjunction_and"),
    ("hoặc", "conjunction_or"),
)


def semantic_risk(reference: str, candidate: str) -> dict:
    """So một cặp dòng, trả {risk, flags}. flags rỗng => coi như chỉ nhiễu OCR."""
    ref = normalize(reference)
    cand = normalize(candidate)
    flags: list[str] = []
    for term, label in _TERMS:
        # đếm số lần xuất hiện thay vì chỉ kiểm tra có/không, bắt được
        # trường hợp thay 1 trong nhiều "và" thành "hoặc".
        if _count_word(ref, term) != _count_word(cand, term):
            flags.append(label)
    if _NUMBER.findall(ref) != _NUMBER.findall(cand):
        flags.append("numeric_or_date_change")
    return {"risk": "critical" if flags else "review", "flags": flags}


def _count_word(text: str, word: str) -> int:
    return len(re.findall(rf"(?<!\w){re.escape(word)}(?!\w)", text))
