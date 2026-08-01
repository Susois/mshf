"""MSHF Inference Pipeline — nhận 2 PDF bất kỳ, trả kết quả đầy đủ.

Sử dụng model đã train từ `train.py` (mshf_label.joblib / mshf_is_tampered.joblib)
cùng full A+B1+C feature pipeline, không phụ thuộc vào VEDTD dataset.

Luồng:
  1. Trích xuất text từ PDF (PyMuPDF hoặc PaddleOCR)
  2. Tính features nhánh A (CER/WER + PhoBERT + LayoutLMv3)
  3. Tính features nhánh B1 (line-level insert/delete/modify)
  4. Tính features nhánh C (geometric từ layout JSON → fallback từ PDF trực tiếp)
  5. Predict bằng model đã train
  6. Localize các dòng bị can thiệp theo trang
  7. Highlight ảnh + xuất report.json + tampered_report.html

Dùng:
  python -m mshf.detect --original orig.pdf --candidate suspicious.pdf --out-dir outputs/detect/doc1
"""
from __future__ import annotations

import argparse
import json
import sys
import unicodedata
import re
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import config  # noqa: E402
from mshf.core.line_align import match_lines, line_cer  # noqa: E402
from mshf.core.semantic_risk import semantic_risk  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalize(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    return re.sub(r"\s+", " ", text).strip().lower()

# ---------------------------------------------------------------------------
# Bước 1: Trích xuất text + bbox từ PDF
# ---------------------------------------------------------------------------

def extract_pages_pymupdf(pdf_path: Path, dpi: int = 150) -> list[list[dict]]:
    """Dùng PyMuPDF để lấy embedded text (nhanh, cho PDF born-digital)."""
    import fitz  # PyMuPDF
    doc = fitz.open(str(pdf_path))
    zoom = dpi / 72.0
    all_pages: list[list[dict]] = []
    for page in doc:
        blocks = page.get_text("dict")["blocks"]
        lines: list[dict] = []
        for block in blocks:
            if block.get("type") != 0:
                continue
            for line in block.get("lines", []):
                text = "".join(s["text"] for s in line.get("spans", []))
                if not text.strip():
                    continue
                bbox = [c * zoom for c in line["bbox"]]
                lines.append({
                    "text": text.strip(),
                    "bbox": [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])],
                    "score": 1.0,
                })
        all_pages.append(lines)
    doc.close()
    return all_pages


def extract_pages_paddleocr(pdf_path: Path, dpi: int = 150) -> list[list[dict]]:
    """Dùng PaddleOCR (cho PDF scan không có embedded text)."""
    import fitz
    import os, uuid
    from paddleocr import PaddleOCR
    ocr = PaddleOCR(lang="vi", use_textline_orientation=False)
    doc = fitz.open(str(pdf_path))
    zoom = dpi / 72.0
    matrix = fitz.Matrix(zoom, zoom)
    all_pages: list[list[dict]] = []
    run_id = uuid.uuid4().hex[:8]
    for i, page in enumerate(doc):
        pix = page.get_pixmap(matrix=matrix, alpha=False)
        tmp = f"_tmp_mshf_{run_id}_{i}.png"
        pix.save(tmp)
        try:
            result = ocr.predict(tmp)
            lines: list[dict] = []
            if result:
                res = result[0]
                for text, score, poly in zip(res["rec_texts"], res["rec_scores"], res["rec_polys"]):
                    xs = [p[0] for p in poly]; ys = [p[1] for p in poly]
                    lines.append({
                        "text": text,
                        "bbox": [float(min(xs)), float(min(ys)), float(max(xs)), float(max(ys))],
                        "score": float(score),
                    })
            all_pages.append(lines)
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)
    doc.close()
    return all_pages

# ---------------------------------------------------------------------------
# Bước 2: Tính features nhánh A (CER/WER + PhoBERT + LayoutLMv3)
# ---------------------------------------------------------------------------

def _compute_cer_wer(orig_pages: list[list[dict]], cand_pages: list[list[dict]]) -> tuple[float, float]:
    orig_text = _normalize(" ".join(l["text"] for p in orig_pages for l in p))
    cand_text = _normalize(" ".join(l["text"] for p in cand_pages for l in p))
    from rapidfuzz.distance import Levenshtein as RFL
    # Cap at 20000 chars to avoid O(n²) timeout on long docs
    MAX_CHARS = 20_000
    o_c = orig_text[:MAX_CHARS]
    c_c = cand_text[:MAX_CHARS]
    cer = RFL.distance(o_c, c_c) / max(len(o_c), 1)
    ow = orig_text.split()
    MAX_WORDS = 3_000
    o_w = " ".join(ow[:MAX_WORDS])
    c_w = " ".join(cand_text.split()[:MAX_WORDS])
    from rapidfuzz.distance import Levenshtein as RFL2
    wer = RFL2.distance(o_w.split(), c_w.split()) / max(len(ow[:MAX_WORDS]), 1)
    return float(min(cer, 1.0)), float(min(wer, 1.0))


def _phobert_embed(lines: list[str], tokenizer, model, device: str = "cpu") -> np.ndarray:
    import torch
    model.eval()
    all_emb: list[np.ndarray] = []
    batch_size = 16
    with torch.no_grad():
        for i in range(0, len(lines), batch_size):
            batch = [t if t.strip() else "." for t in lines[i:i + batch_size]]
            enc = tokenizer(batch, padding=True, truncation=True, max_length=256, return_tensors="pt")
            enc = {k: v.to(device) for k, v in enc.items()}
            out = model(**enc)
            mask = enc["attention_mask"].unsqueeze(-1).float()
            emb = (out.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            all_emb.append(emb.cpu().numpy())
    return np.concatenate(all_emb, axis=0) if all_emb else np.zeros((0, 768))


def _compute_phobert_features(
    orig_pages: list[list[dict]],
    cand_pages: list[list[dict]],
    device: str = "cpu",
) -> dict[str, float]:
    from transformers import AutoTokenizer, AutoModel
    print("  [A] Load PhoBERT (vinai/phobert-base-v2)...")
    tok = AutoTokenizer.from_pretrained("vinai/phobert-base-v2")
    model = AutoModel.from_pretrained("vinai/phobert-base-v2").to(device)

    ref_lines = [l["text"] for p in orig_pages for l in p] or ["."]
    hyp_lines = [l["text"] for p in cand_pages for l in p] or ["."]

    ref_emb = _phobert_embed(ref_lines, tok, model, device)
    hyp_emb = _phobert_embed(hyp_lines, tok, model, device)
    ref_n = ref_emb / (np.linalg.norm(ref_emb, axis=1, keepdims=True) + 1e-9)
    hyp_n = hyp_emb / (np.linalg.norm(hyp_emb, axis=1, keepdims=True) + 1e-9)
    sim = ref_n @ hyp_n.T
    r2h = sim.max(axis=1)
    h2r = sim.max(axis=0)
    combined = np.concatenate([r2h, h2r])
    return {
        "mean_similarity": float(combined.mean()),
        "min_similarity": float(combined.min()),
        "std_similarity": float(combined.std()),
        "ref_to_hyp_mean": float(r2h.mean()),
        "hyp_to_ref_mean": float(h2r.mean()),
    }


def _compute_layoutlmv3_similarity(
    orig_pages: list[list[dict]],
    cand_pages: list[list[dict]],
    orig_images: list[Any],
    cand_images: list[Any],
    device: str = "cpu",
) -> float:
    """Tính LayoutLMv3 cosine similarity trung bình trên các trang."""
    from transformers import LayoutLMv3Processor, LayoutLMv3Model
    import torch
    print("  [A] Load LayoutLMv3 (microsoft/layoutlmv3-base)...")
    proc = LayoutLMv3Processor.from_pretrained("microsoft/layoutlmv3-base", apply_ocr=False)
    lmodel = LayoutLMv3Model.from_pretrained("microsoft/layoutlmv3-base").to(device)
    lmodel.eval()

    sims: list[float] = []
    for pg in range(min(len(orig_pages), len(cand_pages))):
        olines, clines = orig_pages[pg], cand_pages[pg]
        if not olines or not clines:
            continue
        try:
            oimg, cimg = orig_images[pg], cand_images[pg]
            w, h = oimg.size

            def _norm_box(bbox):
                x0, y0, x1, y1 = bbox
                return [max(0, min(1000, int(1000 * x0 / max(w, 1)))),
                        max(0, min(1000, int(1000 * y0 / max(h, 1)))),
                        max(0, min(1000, int(1000 * x1 / max(w, 1)))),
                        max(0, min(1000, int(1000 * y1 / max(h, 1))))]

            def _embed(img, lines):
                words = [l["text"] if l["text"].strip() else "." for l in lines]
                boxes = [_norm_box(l["bbox"]) for l in lines]
                enc = proc(img.convert("RGB"), words, boxes=boxes,
                            return_tensors="pt", truncation=True, padding="max_length")
                enc = {k: v.to(device) for k, v in enc.items()}
                with torch.no_grad():
                    out = lmodel(**enc)
                return out.last_hidden_state.mean(dim=1).squeeze(0).cpu().numpy()

            oe, ce = _embed(oimg, olines), _embed(cimg, clines)
            cos = float(np.dot(oe, ce) / (np.linalg.norm(oe) * np.linalg.norm(ce) + 1e-9))
            sims.append(cos)
        except Exception as e:
            print(f"  [WARN] LayoutLMv3 page {pg}: {e}")
    return float(np.mean(sims)) if sims else 0.0

# ---------------------------------------------------------------------------
# Bước 3: Features nhánh B1 (line-level)
# ---------------------------------------------------------------------------

def _compute_b1_features(
    orig_pages: list[list[dict]],
    cand_pages: list[list[dict]],
) -> dict[str, float]:
    """Tính 16 line-level features (B1) trực tiếp từ PDF pages (không cần .txt file)."""
    ref_lines = [l["text"] for p in orig_pages for l in p]
    cand_lines = [l["text"] for p in cand_pages for l in p]

    from mshf.core.io_utils import as_line_dicts
    pairs = match_lines(as_line_dicts(ref_lines), as_line_dicts(cand_lines), sort_by_y=False)

    insert = delete = modified = match = 0
    mod_cers: list[float] = []
    neg = conj = numeric = critical = 0

    for p in pairs:
        if p["type"] == "inserted":
            insert += 1
        elif p["type"] == "deleted":
            delete += 1
        else:
            cer = p["cer"]
            risk = semantic_risk(p["orig_line"]["text"], p["cand_line"]["text"])
            is_mod = cer > config.CER_MODIFIED_THRESHOLD or bool(risk["flags"])
            if is_mod:
                modified += 1
                mod_cers.append(cer)
                flags = risk["flags"]
                if "negation" in flags: neg += 1
                if any(f.startswith("conjunction") for f in flags): conj += 1
                if "numeric_or_date_change" in flags: numeric += 1
                if flags: critical += 1
            else:
                match += 1

    total = insert + delete + modified + match
    denom = max(total, 1)
    return {
        "ln_insert_count": float(insert),
        "ln_delete_count": float(delete),
        "ln_modified_count": float(modified),
        "ln_match_count": float(match),
        "ln_total_ops": float(insert + delete + modified),
        "ln_insert_ratio": round(insert / denom, 4),
        "ln_delete_ratio": round(delete / denom, 4),
        "ln_modified_ratio": round(modified / denom, 4),
        "ln_unchanged_ratio": round(match / denom, 4),
        "ln_mod_cer_mean": round(sum(mod_cers) / len(mod_cers), 4) if mod_cers else 0.0,
        "ln_mod_cer_min": round(min(mod_cers), 4) if mod_cers else 0.0,
        "ln_mod_cer_max": round(max(mod_cers), 4) if mod_cers else 0.0,
        "ln_negation_count": float(neg),
        "ln_conjunction_count": float(conj),
        "ln_numeric_count": float(numeric),
        "ln_critical_count": float(critical),
    }

# ---------------------------------------------------------------------------
# Bước 4: Features nhánh C (geometric) — fallback từ PDF trực tiếp
# ---------------------------------------------------------------------------

def _pdf_to_layout_dict(pdf_path: Path, dpi: int = 150) -> dict:
    """Chuyển PDF thành dict layout tương tự Tuan5/layout JSON để dùng geometric_features."""
    import fitz
    doc = fitz.open(str(pdf_path))
    pages = []
    for i, page in enumerate(doc):
        pw, ph = page.rect.width, page.rect.height
        blocks = page.get_text("dict")["blocks"]
        lines: list[dict] = []
        for block in blocks:
            if block.get("type") != 0:
                continue
            for line in block.get("lines", []):
                text = "".join(s["text"] for s in line.get("spans", []))
                if not text.strip():
                    continue
                lines.append({"text": text.strip(), "bbox": list(line["bbox"]), "score": 1.0})
        pages.append({"page": i, "page_width": pw, "page_height": ph, "lines": lines})
    doc.close()
    return {"pages": pages}


def _compute_c_features(
    orig_pdf: Path,
    cand_pdf: Path,
) -> dict[str, float]:
    """Tính 16 geometric features (C) trực tiếp từ PDF."""
    from mshf.core.geometric_features import (
        _doc_geo_vector, _GEO_KEYS, _residuals_per_page,
        GEO_RESIDUAL_THRESHOLD, GEOMETRIC_FEATURE_COLS,
    )

    ref_layout = _pdf_to_layout_dict(orig_pdf)
    cand_layout = _pdf_to_layout_dict(cand_pdf)
    feat = {c: 0.0 for c in GEOMETRIC_FEATURE_COLS}

    ref_vec = _doc_geo_vector(ref_layout)
    cand_vec = _doc_geo_vector(cand_layout)
    if ref_vec and cand_vec:
        for k in _GEO_KEYS:
            feat[f"geo_delta_{k}"] = round(abs(cand_vec[k] - ref_vec[k]), 6)

    ref_pages = {p.get("page", i): p for i, p in enumerate(ref_layout.get("pages", []))}
    cand_pages_d = {p.get("page", i): p for i, p in enumerate(cand_layout.get("pages", []))}
    all_residuals: list[float] = []
    for pidx in sorted(set(ref_pages) & set(cand_pages_d)):
        all_residuals.extend(_residuals_per_page(ref_pages[pidx], cand_pages_d[pidx]))

    if all_residuals:
        arr = np.array(all_residuals)
        suspect = int((arr > GEO_RESIDUAL_THRESHOLD).sum())
        feat["geo_residual_mean"] = round(float(arr.mean()), 5)
        feat["geo_residual_max"] = round(float(arr.max()), 5)
        feat["geo_residual_p90"] = round(float(np.percentile(arr, 90)), 5)
        feat["geo_format_suspect_ratio"] = round(suspect / len(arr), 4)
        feat["geo_stable_line_count"] = float(len(arr))
    return feat

# ---------------------------------------------------------------------------
# Bước 5 + 6: Predict + Localize từng dòng
# ---------------------------------------------------------------------------

def _predict(features: dict, model_path: Path) -> dict:
    """Dùng model joblib đã train để predict."""
    import joblib
    artifact = joblib.load(model_path)
    model = artifact["model"]
    labels = artifact["labels"]
    feat_cols = artifact["features"]

    X = np.array([[features.get(c, 0.0) for c in feat_cols]])
    pred_idx = model.predict(X)[0]
    proba = model.predict_proba(X)[0]
    label = labels[pred_idx]

    return {
        "label": label,
        "is_tampered": label != "original",
        "confidence": float(proba[pred_idx]),
        "all_probs": {labels[i]: float(p) for i, p in enumerate(proba)},
    }


def _detect_tampering_pattern(pairs: list[dict]) -> dict:
    """
    Phát hiện pattern của tampering để phân biệt:
    - Real DELETE attack: nhiều deleted lines liên tiếp
    - Real INSERT attack: nhiều inserted lines liên tiếp
    - Real MODIFY attack: scattered modifications với semantic changes
    - PDF regeneration artifact: nhiều low-CER modifications rải rác
    """
    deleted_sequences = []
    inserted_sequences = []
    current_del = []
    current_ins = []
    
    for p in pairs:
        if p["type"] == "deleted":
            current_del.append(p)
            if current_ins and len(current_ins) >= 3:
                inserted_sequences.append(current_ins)
            current_ins = []
        elif p["type"] == "inserted":
            current_ins.append(p)
            if current_del and len(current_del) >= 3:
                deleted_sequences.append(current_del)
            current_del = []
        else:
            if current_del and len(current_del) >= 3:
                deleted_sequences.append(current_del)
            if current_ins and len(current_ins) >= 3:
                inserted_sequences.append(current_ins)
            current_del = []
            current_ins = []
    
    # Final sequences
    if current_del and len(current_del) >= 3:
        deleted_sequences.append(current_del)
    if current_ins and len(current_ins) >= 3:
        inserted_sequences.append(current_ins)
    
    # Analyze modifications
    modified_pairs = [p for p in pairs if p["type"] == "match" and p["cer"] > 0.15]
    low_cer_mods = [p for p in modified_pairs if p["cer"] < 0.35]
    
    total_deleted = sum(len(seq) for seq in deleted_sequences)
    total_inserted = sum(len(seq) for seq in inserted_sequences)
    
    return {
        "deleted_sequences": deleted_sequences,
        "inserted_sequences": inserted_sequences,
        "total_deleted": total_deleted,
        "total_inserted": total_inserted,
        "low_cer_modifications": len(low_cer_mods),
        "total_modifications": len(modified_pairs),
        "is_regenerated_pdf": len(low_cer_mods) > 50 and len(deleted_sequences) == 0,
    }


def _adaptive_cer_threshold(pairs: list[dict], base_threshold: float = config.CER_MODIFIED_THRESHOLD) -> float:
    """
    Điều chỉnh CER threshold động dựa trên phân bố CER trong document.
    
    Nếu phát hiện PDF regeneration (nhiều low-CER modifications):
    → Tăng threshold lên 0.4-0.5 để giảm false positives
    
    Nếu phát hiện real tampering (ít modifications, CER cao):
    → Giữ nguyên threshold 0.2
    """
    modified_cers = [p["cer"] for p in pairs if p["type"] == "match" and p["cer"] > 0.1]
    
    if len(modified_cers) < 10:
        return base_threshold
    
    median_cer = np.median(modified_cers)
    count_low_cer = sum(1 for c in modified_cers if c < 0.35)
    
    # PDF regeneration detected: nhiều modifications với CER thấp
    if len(modified_cers) > 50 and median_cer < 0.3 and count_low_cer > 40:
        print(f"  [ADAPTIVE] Detected PDF regeneration: {len(modified_cers)} mods, median CER={median_cer:.3f}")
        print(f"  [ADAPTIVE] Increasing threshold: 0.2 → 0.4 to reduce false positives")
        return 0.40
    
    # Real tampering: ít modifications, CER cao
    return base_threshold


def _localize_lines(
    orig_pages: list[list[dict]],
    cand_pages: list[list[dict]],
    cer_threshold: float = config.CER_MODIFIED_THRESHOLD,
) -> tuple[list[dict], int]:
    """
    Align document-level (cross-page), trả về list tampered_lines per page
    và tổng số dòng bị can thiệp.

    Đối với INSERT: chỉ đánh dấu dòng thực sự được chèn mới (cer=1.0, orig="").
    Dòng bị đẩy xuống do insert (shifted) KHÔNG bị đánh dấu là tampered.
    
    Đối với DELETE: chỉ đánh dấu dòng thực sự bị xóa (cer=1.0, cand="").
    Dòng có CER thấp do PDF regeneration sẽ bị lọc bỏ bằng adaptive threshold.
    """
    all_orig: list[dict] = []
    for pg, plines in enumerate(orig_pages):
        for ln in plines:
            all_orig.append({**ln, "page_idx": pg})

    all_cand: list[dict] = []
    for pg, plines in enumerate(cand_pages):
        for ln in plines:
            all_cand.append({**ln, "page_idx": pg})

    pairs = match_lines(all_orig, all_cand, sort_by_y=False)
    
    # Detect tampering pattern và adjust threshold
    pattern = _detect_tampering_pattern(pairs)
    print(f"\n[PATTERN ANALYSIS]")
    print(f"  Consecutive deletions: {pattern['total_deleted']} lines in {len(pattern['deleted_sequences'])} sequences")
    print(f"  Consecutive insertions: {pattern['total_inserted']} lines in {len(pattern['inserted_sequences'])} sequences")
    print(f"  Low-CER modifications: {pattern['low_cer_modifications']} (< 0.35)")
    print(f"  Total modifications: {pattern['total_modifications']}")
    if pattern['is_regenerated_pdf']:
        print(f"  ⚠ WARNING: Detected PDF regeneration artifact!")
    
    # Adaptive threshold
    adjusted_threshold = _adaptive_cer_threshold(pairs, cer_threshold)

    n_pages = max(len(orig_pages), len(cand_pages))
    pages_info: list[dict] = [{"page_index": i, "tampered_lines": []} for i in range(n_pages)]
    total_tampered = 0

    for idx, p in enumerate(pairs):
        t = p["type"]
        cer = p["cer"]
        orig_line = p.get("orig_line")
        cand_line = p.get("cand_line")

        # Skip unmodified lines
        if t == "match" and cer <= adjusted_threshold:
            continue

        # Skip low-CER modifications if regenerated PDF detected
        if t == "match" and cer <= 0.35 and pattern['is_regenerated_pdf']:
            continue  # False positive từ PDF regeneration

        # Classify
        if t == "match" and cer > adjusted_threshold:
            risk = semantic_risk(orig_line["text"], cand_line["text"])
            event_type = "modified"
        elif t == "inserted":
            event_type = "inserted"
        elif t == "deleted":
            event_type = "deleted"
        else:
            continue

        # Page for highlighting: prefer cand_line, fallback orig_line
        ref_line = cand_line if cand_line else orig_line
        if ref_line is None:
            continue
        pg = ref_line.get("page_idx", 0)
        if pg >= n_pages:
            continue

        # Lọc các dòng chỉ là số trang / header ngắn (false positive phổ biến)
        cand_text = cand_line["text"].strip() if cand_line else ""
        orig_text = orig_line["text"].strip() if orig_line else ""
        ref_text  = cand_text or orig_text
        # Bỏ qua nếu là dòng số thuần túy (số trang) hoặc quá ngắn (< 3 ký tự)
        if re.match(r'^\d{1,4}$', ref_text) or len(ref_text) < 3:
            continue

        entry = {
            "line_idx": idx,
            "type": event_type,
            "cer": round(cer, 4),
            "orig_text": orig_line["text"] if orig_line else "",
            "cand_text": cand_line["text"] if cand_line else "",
            "bbox": ref_line.get("bbox"),
        }
        pages_info[pg]["tampered_lines"].append(entry)
        total_tampered += 1

    return pages_info, total_tampered

# ---------------------------------------------------------------------------
# Bước 7: Highlight ảnh + xuất HTML
# ---------------------------------------------------------------------------

def _render_pages(pdf_path: Path, dpi: int = 150):
    """Trả list PIL Image của từng trang."""
    import fitz
    from PIL import Image
    doc = fitz.open(str(pdf_path))
    zoom = dpi / 72.0
    matrix = fitz.Matrix(zoom, zoom)
    images = []
    for page in doc:
        pix = page.get_pixmap(matrix=matrix, alpha=False)
        images.append(Image.frombytes("RGB", (pix.width, pix.height), pix.samples))
    doc.close()
    return images


def _highlight_page(image, tampered_lines: list[dict]):
    """Vẽ highlight lên ảnh PIL dựa trên bbox."""
    from PIL import ImageDraw, ImageFont
    draw = ImageDraw.Draw(image, "RGBA")
    try:
        font = ImageFont.truetype("arial.ttf", 14)
    except Exception:
        font = ImageFont.load_default()

    COLOR = {
        "inserted": ((255, 60, 60, 90), (220, 0, 0, 255)),
        "deleted":  ((255, 165, 0, 90), (200, 100, 0, 255)),
        "modified": ((255, 220, 0, 90), (180, 150, 0, 255)),
    }
    for tl in tampered_lines:
        if not tl.get("bbox"):
            continue
        x0, y0, x1, y1 = [int(v) for v in tl["bbox"]]
        fill, outline = COLOR.get(tl["type"], ((200, 0, 200, 90), (200, 0, 200, 255)))
        draw.rectangle([x0, y0, x1, y1], fill=fill, outline=outline, width=2)
        label = tl["type"].upper()
        draw.text((x0, max(0, y0 - 16)), label, fill=outline, font=font)
    return image


def _generate_html(report: dict) -> str:
    verdict_color = "#d4edda" if report["verdict"] == "AUTHENTIC" else "#f8d7da"
    mp = report.get("model_prediction", {})
    hf = report.get("hybrid_features", {})

    model_html = ""
    if mp:
        prob_rows = "".join(
            f"<li>{cls}: {prob:.2%}</li>" for cls, prob in mp.get("all_probs", {}).items()
        )
        model_html = f"""
<div style="padding:14px;background:#e7f3fe;border:1px solid #b6d4fe;border-radius:8px;margin-bottom:18px">
  <h2 style="margin:0;color:#084298">Kết quả model (MSHF A+B1+C)</h2>
  <p><strong>Nhãn dự đoán:</strong> {mp.get('label','?')}<br>
     <strong>Giả mạo:</strong> {'Có' if mp.get('is_tampered') else 'Không'}<br>
     <strong>Confidence:</strong> {mp.get('confidence', 0):.2%}</p>
  <p><strong>Xác suất từng lớp:</strong></p><ul>{prob_rows}</ul>
</div>"""

    feat_html = ""
    if hf:
        feat_rows = "".join(f"<li><strong>{k}:</strong> {v:.6f}</li>" for k, v in hf.items())
        feat_html = f"""
<div style="padding:14px;background:#f0f0f0;border-radius:8px;margin-bottom:18px">
  <h2 style="margin:0">Features (A+B1+C — {len(hf)} features)</h2>
  <ul style="columns:2">{feat_rows}</ul>
</div>"""

    pages_html = ""
    for pg in report["pages"]:
        img_file = f"page_{pg['page_index']:03d}_highlighted.png"
        tlines = pg["tampered_lines"]
        if tlines:
            rows = "".join(
                f"<tr style='background:{'#f8d7da' if t['type']=='inserted' else '#fff3cd' if t['type']=='modified' else '#ffeeba'}'>"
                f"<td>{t['line_idx']}</td><td><b>{t['type'].upper()}</b></td>"
                f"<td>{t['cer']:.2%}</td>"
                f"<td style='max-width:300px;word-break:break-all'>{t.get('orig_text','')[:120]}</td>"
                f"<td style='max-width:300px;word-break:break-all'>{t.get('cand_text','')[:120]}</td></tr>"
                for t in tlines
            )
            pages_html += f"""
<h2>Trang {pg['page_index'] + 1} — {len(tlines)} dòng bị can thiệp</h2>
<table border='1' cellpadding='5' style='border-collapse:collapse;width:100%;font-size:12px'>
<tr><th>#</th><th>Loại</th><th>CER</th><th>Văn bản gốc</th><th>Văn bản nghi vấn</th></tr>
{rows}</table>
<img src='{img_file}' style='width:100%;border:1px solid #ccc;margin-top:8px'/>"""
        else:
            pages_html += f"<h2>Trang {pg['page_index'] + 1}</h2><p style='color:green'>✓ Không phát hiện can thiệp.</p>"

    return f"""<!DOCTYPE html>
<html lang='vi'><head><meta charset='UTF-8'>
<title>MSHF Detection Report</title>
<style>body{{font-family:Arial,sans-serif;padding:20px;max-width:1200px;margin:auto}}
h1{{color:#1F4E79}}h2{{color:#2e75b6;margin-top:28px}}table{{font-size:12px}}</style>
</head><body>
<h1>🔍 MSHF — Báo cáo phát hiện giả mạo tài liệu</h1>
<div style='padding:14px;background:{verdict_color};border-radius:8px;margin-bottom:18px'>
  <h2 style='margin:0'>Kết quả: {report['verdict']}</h2>
  <p>PDF gốc: <code>{report['original_pdf']}</code><br>
     PDF nghi vấn: <code>{report['candidate_pdf']}</code><br>
     Tổng dòng bị can thiệp: <strong>{report['total_tampered_lines']}</strong> / {report['total_lines']}</p>
</div>
{model_html}
{feat_html}
{pages_html}
</body></html>"""

# ---------------------------------------------------------------------------
# Pipeline chính: detect()
# ---------------------------------------------------------------------------

def detect(
    original_pdf: Path,
    candidate_pdf: Path,
    out_dir: Path,
    model_path: Path | None = None,
    mode: str = "auto",
    ocr_backend: str = "pymupdf",
    dpi: int = 150,
    device: str = "auto",
    cer_threshold: float = config.CER_MODIFIED_THRESHOLD,
) -> dict:
    """
    Phân tích 1 cặp PDF (gốc + nghi vấn), trả về report dict và lưu output.

    Args:
        original_pdf:    PDF gốc tham chiếu
        candidate_pdf:   PDF nghi vấn
        out_dir:         Thư mục lưu output (ảnh highlight, JSON, HTML)
        model_path:      Đường dẫn tới mshf_label.joblib (None = tự tìm)
        mode:            "auto" (tự chọn), "fast" (B1+C only), "full" (A+B1+C)
        ocr_backend:     "pymupdf" (nhanh) hoặc "paddleocr" (cho scan)
        dpi:             DPI render PDF sang ảnh
        device:          "auto" (tự phát hiện GPU), "cpu", "cuda"
        cer_threshold:   Ngưỡng CER để coi một dòng là "bị sửa"
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # Auto-detect device
    if device == "auto":
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"

    # Auto-detect mode
    if mode == "auto":
        # Nếu có GPU → full, nếu không → fast
        mode = "full" if device == "cuda" else "fast"

    skip_phobert = (mode == "fast")
    skip_layoutlmv3 = (mode == "fast")

    # Tìm model path
    if model_path is None:
        candidates = [
            Path(__file__).resolve().parent.parent / "outputs" / "training" / "mshf_label.joblib",
            config.OUTPUT_DIR / "training" / "mshf_label.joblib",
        ]
        for c in candidates:
            if c.exists():
                model_path = c
                break

    # --- Bước 1: Trích xuất text ---
    print(f"\n[1/5] Trích xuất text ({ocr_backend})...")
    extract_fn = extract_pages_paddleocr if ocr_backend == "paddleocr" else extract_pages_pymupdf
    orig_pages = extract_fn(original_pdf, dpi)
    cand_pages = extract_fn(candidate_pdf, dpi)
    print(f"  Original: {len(orig_pages)} trang, {sum(len(p) for p in orig_pages)} dòng")
    print(f"  Candidate: {len(cand_pages)} trang, {sum(len(p) for p in cand_pages)} dòng")

    # --- Bước 2: Features nhánh A ---
    use_full = (mode == "full")
    print(f"\n[2/5] Tính features A (doc-level) — mode={mode})...")
    cer, wer = _compute_cer_wer(orig_pages, cand_pages)
    print(f"  CER={cer:.4f}, WER={wer:.4f}")

    # Luôn render ảnh (cần cho highlight và LayoutLMv3)
    orig_images = _render_pages(original_pdf, dpi)
    cand_images = _render_pages(candidate_pdf, dpi)

    if use_full:
        pho_feats = _compute_phobert_features(orig_pages, cand_pages, device)
        lv3_sim = _compute_layoutlmv3_similarity(orig_pages, cand_pages, orig_images, cand_images, device)
    else:
        # fast mode: B1+C đủ để phân biệt insert/delete/modify, không cần model nặng
        pho_feats = {"mean_similarity": 0.0, "min_similarity": 0.0,
                     "std_similarity": 0.0, "ref_to_hyp_mean": 0.0, "hyp_to_ref_mean": 0.0}
        lv3_sim = 0.0
        print("  [fast mode] Bỏ qua PhoBERT và LayoutLMv3 — dùng B1+C features")

    a_feats: dict[str, float] = {"cer": cer, "wer": wer, **pho_feats, "layoutlmv3_cosine_similarity": lv3_sim}

    # --- Bước 3: Features nhánh B1 ---
    print(f"\n[3/5] Tính features B1 (line-level)...")
    b1_feats = _compute_b1_features(orig_pages, cand_pages)
    print(f"  insert={b1_feats['ln_insert_count']:.0f}, delete={b1_feats['ln_delete_count']:.0f}, "
          f"modified={b1_feats['ln_modified_count']:.0f}")

    # --- Bước 4: Features nhánh C ---
    print(f"\n[4/5] Tính features C (geometric)...")
    c_feats = _compute_c_features(original_pdf, candidate_pdf)

    all_features = {**a_feats, **b1_feats, **c_feats}

    # --- Bước 5: Predict + Localize ---
    print(f"\n[5/5] Dự đoán...")
    prediction = None
    if model_path and model_path.exists():
        prediction = _predict(all_features, model_path)
        print(f"  Label: {prediction['label']} | Confidence: {prediction['confidence']:.2%}")
        print(f"  Xác suất: " + " | ".join(f"{k}={v:.1%}" for k, v in prediction["all_probs"].items()))

        # --- Override khi model underconfident nhưng line-level evidence rõ ràng ---
        # Trường hợp: model nói "original" nhưng B1 phát hiện có dòng bị sửa thực sự
        mod_count = b1_feats.get("ln_modified_count", 0)
        ins_count = b1_feats.get("ln_insert_count", 0)
        del_count = b1_feats.get("ln_delete_count", 0)
        critical  = b1_feats.get("ln_critical_count", 0)
        mod_cer   = b1_feats.get("ln_mod_cer_mean", 0)

        if prediction["label"] == "original":
            # Có insert/delete thực sự (CER=1.0) → chắc chắn bị tấn công
            if ins_count >= 1:
                print(f"  [OVERRIDE] {ins_count} dòng inserted → label: insert")
                prediction["label"] = "insert"
                prediction["is_tampered"] = True
            elif del_count >= 1:
                print(f"  [OVERRIDE] {del_count} dòng deleted → label: delete")
                prediction["label"] = "delete"
                prediction["is_tampered"] = True
            # Có modify với semantic change thực sự (không phải nhiễu OCR)
            elif mod_count >= 1 and critical >= 1 and mod_cer >= 0.3:
                print(f"  [OVERRIDE] {mod_count} dòng modified (critical={critical}, cer={mod_cer:.2f}) → label: modify")
                prediction["label"] = "modify"
                prediction["is_tampered"] = True
    else:
        print("  [WARN] Không tìm thấy model — chỉ chạy localization")

    pages_info, total_tampered = _localize_lines(orig_pages, cand_pages, cer_threshold)
    total_lines = sum(len(p) for p in orig_pages)

    # Highlight ảnh
    for pg_info in pages_info:
        pg_idx = pg_info["page_index"]
        img = (cand_images[pg_idx].copy() if pg_idx < len(cand_images) else None)
        if img is None:
            from PIL import Image as PILImage
            img = PILImage.new("RGB", (800, 200), "white")
        if pg_info["tampered_lines"]:
            img = _highlight_page(img, pg_info["tampered_lines"])
        img_path = out_dir / f"page_{pg_idx:03d}_highlighted.png"
        img.save(img_path)
        pg_info["image_file"] = str(img_path)

    report = {
        "original_pdf": str(original_pdf),
        "candidate_pdf": str(candidate_pdf),
        "total_lines": total_lines,
        "total_tampered_lines": total_tampered,
        "verdict": "AUTHENTIC" if total_tampered == 0 else "TAMPERED",
        "pages": pages_info,
    }
    if prediction:
        report["model_prediction"] = prediction
    report["hybrid_features"] = all_features

    # Lưu JSON
    json_path = out_dir / "report.json"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    # Lưu HTML
    html_path = out_dir / "tampered_report.html"
    html_path.write_text(_generate_html(report), encoding="utf-8")

    print(f"\n{'='*60}")
    print(f"KẾT QUẢ: {report['verdict']}")
    print(f"Mode     : {mode} | Device: {device}")
    if prediction:
        print(f"Nhãn dự đoán : {prediction['label']}")
        print(f"Confidence   : {prediction['confidence']:.2%}")
    print(f"Dòng can thiệp: {total_tampered} / {total_lines}")
    print(f"Report JSON  : {json_path}")
    print(f"Report HTML  : {html_path}")
    print(f"{'='*60}\n")

    return report

# ---------------------------------------------------------------------------
# main() — CLI
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(
        description="MSHF — Phát hiện giả mạo tài liệu PDF tiếng Việt",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Cách dùng đơn giản nhất (tự động chọn config tốt nhất):
  python -m mshf.detect --original goc.pdf --candidate nghi_van.pdf --out-dir outputs/detect/ket_qua

Với PDF scan (không có text nhúng):
  python -m mshf.detect --original goc.pdf --candidate nghi_van.pdf --out-dir outputs/detect/ket_qua --ocr paddleocr

Chế độ nâng cao:
  --mode fast  : Nhanh nhất, chỉ dùng B1+C features (~5 giây, CPU)
  --mode full  : Đầy đủ A+B1+C, cần tải PhoBERT lần đầu (~2 phút, CPU hoặc ~30 giây GPU)
  --mode auto  : Tự chọn (full nếu có GPU, fast nếu không) ← MẶC ĐỊNH
""")
    ap.add_argument("--original", required=True, type=Path, help="PDF gốc tham chiếu")
    ap.add_argument("--candidate", required=True, type=Path, help="PDF nghi vấn")
    ap.add_argument("--out-dir", type=Path, default=Path("outputs/detect/result"), help="Thư mục output")
    ap.add_argument("--model", type=Path, default=None, help="Đường dẫn tới mshf_label.joblib (tự tìm nếu không chỉ định)")
    ap.add_argument("--mode", choices=["auto", "fast", "full"], default="full",
                    help="Chế độ: full (A+B1+C, mặc định), fast (B1+C only, không cần internet), auto (tự chọn theo GPU)")
    ap.add_argument("--ocr", choices=["pymupdf", "paddleocr"], default="pymupdf",
                    help="Backend OCR: pymupdf (nhanh, cho PDF có text) hoặc paddleocr (cho PDF scan)")
    ap.add_argument("--dpi", type=int, default=150, help="DPI render ảnh (default=150)")
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto",
                    help="Device: auto (tự phát hiện GPU), cpu, cuda")
    ap.add_argument("--cer-threshold", type=float, default=config.CER_MODIFIED_THRESHOLD,
                    help=f"Ngưỡng CER đánh dấu dòng bị sửa (default={config.CER_MODIFIED_THRESHOLD})")
    args = ap.parse_args()

    if not args.original.exists():
        print(f"[ERROR] Không tìm thấy: {args.original}", file=sys.stderr)
        return 1
    if not args.candidate.exists():
        print(f"[ERROR] Không tìm thấy: {args.candidate}", file=sys.stderr)
        return 1

    detect(
        original_pdf=args.original,
        candidate_pdf=args.candidate,
        out_dir=args.out_dir,
        model_path=args.model,
        mode=args.mode,
        ocr_backend=args.ocr,
        dpi=args.dpi,
        device=args.device,
        cer_threshold=args.cer_threshold,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
