"""Đọc dữ liệu VEDTD (OCR text, ground-truth text, layout JSON) và chuẩn hoá.

Chỉ đọc từ Tuan1_2 / Tuan5; không sửa các folder nguồn. Mọi hàm ở đây trả về
cấu trúc thuần Python để các module feature dùng lại.
"""
from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402


_WHITESPACE = re.compile(r"\s+")


def normalize(text: str) -> str:
    """NFC + gộp khoảng trắng + lowercase. Dùng cho so khớp text-diff."""
    text = unicodedata.normalize("NFC", text)
    return _WHITESPACE.sub(" ", text).strip().lower()


def read_text_lines(path: Path) -> list[str]:
    """Đọc file .txt, mỗi dòng logic là một phần tử; bỏ dòng trắng."""
    if not path.exists():
        return []
    raw = path.read_text(encoding="utf-8", errors="replace")
    return [ln.strip() for ln in raw.splitlines() if ln.strip()]


def ocr_text_path(category: str, doc_id: str) -> Path:
    return config.OCR_TEXT_ROOT / category / f"{doc_id}.txt"


def gt_text_path(category: str, doc_id: str) -> Path:
    return config.GT_TEXT_ROOT / category / f"{doc_id}.txt"


def layout_json_path(category: str, doc_id: str) -> Path:
    return config.LAYOUT_JSON_ROOT / category / f"{doc_id}.json"


def load_layout_json(path: Path) -> dict | None:
    """Đọc layout JSON PaddleOCR: {pdf_path, num_pages, avg_rec_score, pages[]}."""
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def layout_lines_flat(layout: dict) -> list[dict]:
    """Trải phẳng tất cả dòng của mọi trang thành list {text, bbox, page, score}."""
    out: list[dict] = []
    for page in layout.get("pages", []):
        page_idx = page.get("page", 0)
        for line in page.get("lines", []):
            bbox = line.get("bbox") or line.get("box")
            if bbox and line.get("text", "").strip():
                out.append({
                    "text": line["text"],
                    "bbox": list(bbox),
                    "page": page_idx,
                    "score": line.get("score", 1.0),
                })
    return out


def discover_doc_ids(category: str = config.ORIGINAL_CAT) -> list[str]:
    """Liệt kê doc_id (stem file .txt) có trong một category của OCR output."""
    cat_dir = config.OCR_TEXT_ROOT / category
    if not cat_dir.exists():
        return []
    return sorted(p.stem for p in cat_dir.glob("*.txt"))


def as_line_dicts(texts: list[str]) -> list[dict]:
    """Bọc list text thành list dict {text, bbox} cho line_align (bbox giả theo thứ tự)."""
    return [{"text": t, "bbox": [0.0, float(i), 1.0, float(i) + 1.0]} for i, t in enumerate(texts)]
