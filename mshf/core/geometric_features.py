"""Đặc trưng HÌNH HỌC mức tài liệu + residual sau căn chỉnh dòng ổn định.

Nhánh C trong MSHF. Port logic 11-dim geometric của đề tài gốc
(Tuan5/4.layout_ocr/layout_features.py) sang MSHF để độc lập, rồi bổ sung:
  - delta 11-dim giữa candidate và original (bắt tấn công 5.layout: font/spacing/margin),
  - geometric residual: căn chỉnh (scale+translate) từ các dòng KHỚP, đo lệch vị trí
    còn lại -> tách reflow (do insert/delete/modify) khỏi format-manipulation thật.
"""
from __future__ import annotations

from statistics import median

import numpy as np

from mshf.core import io_utils
from mshf.core.line_align import match_lines
import config

# 11-dim geometric keys (port nguyên từ layout_features.py của đề tài gốc).
_GEO_KEYS = [
    "n_lines", "text_density", "avg_font_height", "std_font_height",
    "avg_line_spacing", "std_line_spacing", "margin_left", "margin_top",
    "margin_right", "margin_bottom", "y_center_std",
]

GEOMETRIC_FEATURE_COLS = (
    [f"geo_delta_{k}" for k in _GEO_KEYS]
    + ["geo_residual_mean", "geo_residual_max", "geo_residual_p90",
       "geo_format_suspect_ratio", "geo_stable_line_count"]
)

GEO_RESIDUAL_THRESHOLD = 0.018  # lệch > 1.8% đường chéo trang => nghi format thật


def _zeros() -> dict:
    return {c: 0.0 for c in GEOMETRIC_FEATURE_COLS}


def _page_features(page: dict) -> dict | None:
    lines = [l for l in page.get("lines", []) if l.get("bbox")]
    page_w = page.get("page_width")
    page_h = page.get("page_height")
    if not lines or not page_w or not page_h:
        return None
    heights, y_centers = [], []
    x0s, y0s, x1s, y1s = [], [], [], []
    total_area = 0.0
    for line in lines:
        x0, y0, x1, y1 = line["bbox"]
        heights.append(y1 - y0)
        y_centers.append((y0 + y1) / 2)
        x0s.append(x0); y0s.append(y0); x1s.append(x1); y1s.append(y1)
        total_area += max(0.0, x1 - x0) * max(0.0, y1 - y0)
    y_sorted = sorted(y_centers)
    spacings = np.diff(y_sorted) if len(y_sorted) > 1 else np.array([0.0])
    return {
        "n_lines": len(lines),
        "text_density": total_area / (page_w * page_h),
        "avg_font_height": float(np.mean(heights)) / page_h,
        "std_font_height": float(np.std(heights)) / page_h,
        "avg_line_spacing": float(np.mean(spacings)) / page_h,
        "std_line_spacing": float(np.std(spacings)) / page_h,
        "margin_left": min(x0s) / page_w,
        "margin_top": min(y0s) / page_h,
        "margin_right": (page_w - max(x1s)) / page_w,
        "margin_bottom": (page_h - max(y1s)) / page_h,
        "y_center_std": float(np.std(y_centers)) / page_h,
    }


def _doc_geo_vector(layout: dict) -> dict | None:
    pages = layout.get("pages", [])
    feats = [_page_features(p) for p in pages]
    feats = [f for f in feats if f is not None]
    if not feats:
        return None
    keys = [k for k in _GEO_KEYS if k != "n_lines"]
    agg = {k: float(np.mean([f[k] for f in feats])) for k in keys}
    agg["n_lines"] = int(sum(f["n_lines"] for f in feats))
    return agg


def _centre(bbox: list[float]) -> tuple[float, float]:
    return (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2


def _residuals_per_page(ref_page: dict, cand_page: dict) -> list[float]:
    """Căn chỉnh scale+translate từ dòng khớp, trả residual chuẩn hoá của các dòng khớp."""
    page_w = cand_page.get("page_width") or ref_page.get("page_width")
    page_h = cand_page.get("page_height") or ref_page.get("page_height")
    if not page_w or not page_h:
        return []
    ref_lines = [l for l in ref_page.get("lines", []) if l.get("bbox") and l.get("text", "").strip()]
    cand_lines = [l for l in cand_page.get("lines", []) if l.get("bbox") and l.get("text", "").strip()]
    if not ref_lines or not cand_lines:
        return []

    pairs = match_lines(ref_lines, cand_lines, sort_by_y=True)
    stable = [p for p in pairs if p["type"] == "match" and p["cer"] <= config.CER_MODIFIED_THRESHOLD]
    if len(stable) < 2:
        return []

    # Ước lượng scale theo bề rộng/cao bbox (robust bằng median).
    sx = median([
        (p["orig_line"]["bbox"][2] - p["orig_line"]["bbox"][0]) /
        max(p["cand_line"]["bbox"][2] - p["cand_line"]["bbox"][0], 1e-6)
        for p in stable
    ])
    sy = median([
        (p["orig_line"]["bbox"][3] - p["orig_line"]["bbox"][1]) /
        max(p["cand_line"]["bbox"][3] - p["cand_line"]["bbox"][1], 1e-6)
        for p in stable
    ])
    offs = []
    for p in stable:
        rx, ry = _centre(p["orig_line"]["bbox"])
        cx, cy = _centre(p["cand_line"]["bbox"])
        offs.append((rx - sx * cx, ry - sy * cy))
    tx = median([o[0] for o in offs])
    ty = median([o[1] for o in offs])

    diag = float(np.hypot(page_w, page_h))
    residuals = []
    for p in stable:
        rx, ry = _centre(p["orig_line"]["bbox"])
        cx, cy = _centre(p["cand_line"]["bbox"])
        px, py = sx * cx + tx, sy * cy + ty
        residuals.append(float(np.hypot(rx - px, ry - py)) / diag)
    return residuals


def geometric_features(doc_id: str, category: str) -> dict:
    """Đặc trưng hình học cho cặp (1.original, category) cùng doc_id từ layout JSON."""
    ref_layout = io_utils.load_layout_json(io_utils.layout_json_path(config.ORIGINAL_CAT, doc_id))
    cand_layout = io_utils.load_layout_json(io_utils.layout_json_path(category, doc_id))
    feat = _zeros()
    if ref_layout is None or cand_layout is None:
        return feat

    ref_vec = _doc_geo_vector(ref_layout)
    cand_vec = _doc_geo_vector(cand_layout)
    if ref_vec and cand_vec:
        for k in _GEO_KEYS:
            feat[f"geo_delta_{k}"] = round(abs(cand_vec[k] - ref_vec[k]), 6)

    # Residual theo trang ghép index (đơn điệu theo thứ tự trang).
    ref_pages = {p.get("page", i): p for i, p in enumerate(ref_layout.get("pages", []))}
    cand_pages = {p.get("page", i): p for i, p in enumerate(cand_layout.get("pages", []))}
    all_residuals: list[float] = []
    for pidx in sorted(set(ref_pages) & set(cand_pages)):
        all_residuals.extend(_residuals_per_page(ref_pages[pidx], cand_pages[pidx]))

    if all_residuals:
        arr = np.array(all_residuals)
        suspect = int((arr > GEO_RESIDUAL_THRESHOLD).sum())
        feat["geo_residual_mean"] = round(float(arr.mean()), 5)
        feat["geo_residual_max"] = round(float(arr.max()), 5)
        feat["geo_residual_p90"] = round(float(np.percentile(arr, 90)), 5)
        feat["geo_format_suspect_ratio"] = round(suspect / len(arr), 4)
        feat["geo_stable_line_count"] = len(arr)
    return feat
