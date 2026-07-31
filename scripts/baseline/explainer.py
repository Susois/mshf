# explainer.py
"""
Phần 8 – Output & Explanation
Input : 2 PDF (gốc + nghi vấn)
Output:
  - tampered_report.json  : trang nào, dòng nào, CER dòng đó, text gốc vs text nghi vấn
  - output_images/        : ảnh từng trang có highlight bbox dòng bị sửa (màu đỏ)
  - tampered_report.html  : báo cáo trực quan, kèm ảnh inline

Hỗ trợ 2 backend trích xuất text:
  - pymupdf (mặc định) : nhanh, dùng embedded text từ PDF (born-digital)
  - paddleocr           : chậm hơn, dùng OCR (phù hợp PDF scan)
"""
import argparse, json, pickle, re, unicodedata, sys, io
from pathlib import Path

# Fix Unicode encoding on Windows
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import fitz                          # PyMuPDF
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from rapidfuzz.distance import Levenshtein as RFLevenshtein

# Thứ tự đặc trưng PHẢI khớp chính xác với lúc train (xem baseline_proposed.py / train_hybrid_fusion.py)
FEATURE_ORDER = [
    "cer", "wer",
    "mean_similarity", "min_similarity", "std_similarity",
    "ref_to_hyp_mean", "hyp_to_ref_mean",
    "layoutlmv3_cosine_similarity",
]

# Hằng số CER threshold để phát hiện dòng bị chỉnh sửa
CER_THRESHOLD = 0.2

# ============================
# 1. Render PDF → ảnh
# ============================
def render_pages(pdf_path: Path, dpi: int = 150) -> list[dict]:
    doc = fitz.open(pdf_path)
    pages = []
    zoom = dpi / 72.0
    matrix = fitz.Matrix(zoom, zoom)
    for i, page in enumerate(doc):
        pix = page.get_pixmap(matrix=matrix, alpha=False)
        img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        pages.append({"page": i, "image": img, "width": pix.width, "height": pix.height})
    doc.close()
    return pages

# ============================
# 2a. OCR bằng PaddleOCR (chậm, phù hợp PDF scan)
# ============================
def ocr_pages(pdf_path: Path, dpi: int = 150) -> list[list[dict]]:
    from paddleocr import PaddleOCR
    import os, uuid
    ocr = PaddleOCR(lang="vi", use_textline_orientation=False)
    doc = fitz.open(pdf_path)
    zoom = dpi / 72.0
    matrix = fitz.Matrix(zoom, zoom)
    all_pages = []
    run_id = uuid.uuid4().hex[:8]
    for i, page in enumerate(doc):
        pix = page.get_pixmap(matrix=matrix, alpha=False)
        img_path = f"_tmp_page_{run_id}_{i}.png"
        pix.save(img_path)
        try:
            result = ocr.predict(img_path)
            lines = []
            if result:
                res = result[0]
                texts = res["rec_texts"]
                scores = res["rec_scores"]
                polys = res["rec_polys"]
                for text, score, poly in zip(texts, scores, polys):
                    xs = [p[0] for p in poly]; ys = [p[1] for p in poly]
                    lines.append({
                        "line_idx": len(lines),
                        "text": text,
                        "score": float(score),
                        "bbox": [float(min(xs)), float(min(ys)), float(max(xs)), float(max(ys))],
                    })
            all_pages.append(lines)
        finally:
            if os.path.exists(img_path):
                os.remove(img_path)
    doc.close()
    return all_pages

# ============================
# 2b. Trích xuất text bằng PyMuPDF (nhanh, dùng embedded text)
# ============================
def extract_text_pymupdf(pdf_path: Path, dpi: int = 150) -> list[list[dict]]:
    """
    Trích xuất text có bbox từ PDF sử dụng PyMuPDF (fitz).
    Nhanh hơn PaddleOCR rất nhiều vì đọc trực tiếp embedded text.
    Bbox được scale theo DPI để khớp với ảnh render.
    """
    doc = fitz.open(pdf_path)
    zoom = dpi / 72.0
    all_pages = []
    for page in doc:
        blocks = page.get_text('dict')['blocks']
        lines = []
        for block in blocks:
            if block['type'] != 0:  # chỉ lấy text block
                continue
            for line in block.get('lines', []):
                spans = line.get('spans', [])
                text = ''.join(s['text'] for s in spans)
                if not text.strip():
                    continue
                # Scale bbox theo DPI
                bbox = [c * zoom for c in line['bbox']]
                lines.append({
                    "line_idx": len(lines),
                    "text": text.strip(),
                    "score": 1.0,
                    "bbox": [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])],
                })
        all_pages.append(lines)
    doc.close()
    return all_pages

# ============================
# 3. Tính CER từng dòng riêng lẻ
# ============================
def normalize(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    return re.sub(r"\s+", " ", text).strip().lower()

def edit_distance(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for ca in a:
        curr = [prev[0] + 1]
        for j, cb in enumerate(b):
            curr.append(min(curr[j] + 1, prev[j + 1] + 1, prev[j] + (ca != cb)))
        prev = curr
    return prev[-1]

def line_cer(ref: str, hyp: str) -> float:
    r, h = normalize(ref), normalize(hyp)
    if not r: return 0.0 if not h else 1.0
    return edit_distance(r, h) / len(r)

# ============================
# 4. Match dòng gốc vs nghi vấn (DP alignment)
# ============================
def match_lines(orig_lines: list[dict], cand_lines: list[dict], sort_by_y: bool = True) -> list[dict]:
    """
    Căn chỉnh dòng bằng Dynamic Programming (giống sequence alignment).
    Với insert/delete, độ dài 2 danh sách có thể khác nhau.
    sort_by_y=False khi dòng đã ở đúng thứ tự đọc (ví dụ cross-page alignment).
    Sử dụng banded DP để tối ưu tốc độ cho tài liệu dài.
    """
    if sort_by_y:
        orig_sorted = sorted(orig_lines, key=lambda l: l["bbox"][1])
        cand_sorted = sorted(cand_lines, key=lambda l: l["bbox"][1])
    else:
        orig_sorted = list(orig_lines)
        cand_sorted = list(cand_lines)

    n, m = len(orig_sorted), len(cand_sorted)
    
    # Banded DP: only compute within band_width of diagonal
    # Band width adapts to size difference + margin
    band_width = max(50, abs(n - m) * 3 + 20)
    
    INF = float("inf")
    # Use dict for sparse DP to save memory on large inputs
    dp = {}
    dp[(0, 0)] = 0
    # Also need to handle edge: dp[i][0] = i, dp[0][j] = j (within band)
    for i in range(1, n + 1):
        if i <= band_width:
            dp[(i, 0)] = float(i)
    for j in range(1, m + 1):
        if j <= band_width:
            dp[(0, j)] = float(j)
    
    # Cache CER computations
    cer_cache = {}
    def cached_line_cer(i_idx, j_idx):
        key = (i_idx, j_idx)
        if key not in cer_cache:
            cer_cache[key] = line_cer(orig_sorted[i_idx]["text"], cand_sorted[j_idx]["text"])
        return cer_cache[key]
    
    for i in range(1, n + 1):
        # Compute the band range for j
        diag_j = int(i * m / n) if n > 0 else i
        j_lo = max(1, diag_j - band_width)
        j_hi = min(m, diag_j + band_width)
        for j in range(j_lo, j_hi + 1):
            cost = INF
            # Match/substitute
            if (i-1, j-1) in dp:
                cer = cached_line_cer(i-1, j-1)
                cost = min(cost, dp[(i-1, j-1)] + cer)
            # Delete from orig (orig line not in cand)
            if (i-1, j) in dp:
                cost = min(cost, dp[(i-1, j)] + 1.0)
            # Insert into cand (cand line not in orig)
            if (i, j-1) in dp:
                cost = min(cost, dp[(i, j-1)] + 1.0)
            if cost < INF:
                dp[(i, j)] = cost

    # Traceback
    pairs = []
    i, j = n, m
    while i > 0 or j > 0:
        matched = False
        if i > 0 and j > 0 and (i-1, j-1) in dp and (i, j) in dp:
            cer = cached_line_cer(i-1, j-1)
            if abs(dp[(i, j)] - (dp[(i-1, j-1)] + cer)) < 1e-9:
                pairs.append({
                    "orig_line": orig_sorted[i-1],
                    "cand_line": cand_sorted[j-1],
                    "cer": cer,
                    "type": "match",
                })
                i -= 1; j -= 1
                matched = True
        if not matched and i > 0 and (i-1, j) in dp and (i, j) in dp:
            if abs(dp[(i, j)] - (dp[(i-1, j)] + 1.0)) < 1e-9:
                pairs.append({"orig_line": orig_sorted[i-1], "cand_line": None, "cer": 1.0, "type": "deleted"})
                i -= 1
                matched = True
        if not matched and j > 0:
            pairs.append({"orig_line": None, "cand_line": cand_sorted[j-1], "cer": 1.0, "type": "inserted"})
            j -= 1

    pairs.reverse()
    return pairs

# ============================
# 5. Highlight bbox trên ảnh
# ============================
def highlight_page(image: Image.Image, tampered_pairs: list[dict]) -> Image.Image:
    draw = ImageDraw.Draw(image, "RGBA")
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except Exception:
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
        except Exception:
            font = ImageFont.load_default()

    COLOR_MAP = {
        "inserted": (255, 80,  80,  80),   # đỏ – dòng chèn thêm
        "deleted":  (255, 165,  0,  80),   # cam – dòng bị xóa (không còn trong cand)
        "modified": (255, 200,  0,  80),   # vàng – dòng bị sửa
    }
    BORDER_MAP = {
        "inserted": (220, 0,   0,  255),
        "deleted":  (200, 100, 0,  255),
        "modified": (180, 150, 0,  255),
    }

    for pair in tampered_pairs:
        t = pair["type"]
        line = pair.get("cand_line") or pair.get("orig_line")
        if not line: continue
        x0, y0, x1, y1 = [int(v) for v in line["bbox"]]
        draw.rectangle([x0, y0, x1, y1], fill=COLOR_MAP.get(t, (200,0,0,80)), outline=BORDER_MAP.get(t, (200,0,0,255)), width=2)
        label = {"inserted": "INSERTED", "deleted": "DELETED", "modified": "MODIFIED"}.get(t, t)
        draw.text((x0, max(0, y0 - 18)), label, fill=BORDER_MAP.get(t, (200,0,0,255)), font=font)

    return image

# ============================
# 5b. Trích xuất 8 đặc trưng hybrid (dùng chung cho explainer.py và detector.py)
# ============================

def compute_document_cer_wer(ref_text: str, hyp_text: str) -> tuple[float, float]:
    """Tính CER và WER của toàn bộ tài liệu"""
    ref = normalize(ref_text)
    hyp = normalize(hyp_text)
    ref_words = ref.split()
    hyp_words = hyp.split()
    cer = RFLevenshtein.distance(ref, hyp) / max(len(ref), 1)
    wer = RFLevenshtein.distance(ref_words, hyp_words) / max(len(ref_words), 1)
    return cer, wer


def _phobert_embed(lines: list[str], tokenizer, model, device: str = "cpu", batch_size: int = 16) -> np.ndarray:
    """Tính embedding PhoBERT cho danh sách dòng văn bản"""
    import torch
    model.eval()
    all_emb = []
    with torch.no_grad():
        for i in range(0, len(lines), batch_size):
            batch = [t if t.strip() else "." for t in lines[i:i + batch_size]]
            enc = tokenizer(batch, padding=True, truncation=True, max_length=256, return_tensors="pt").to(device)
            out = model(**enc)
            mask = enc["attention_mask"].unsqueeze(-1).float()
            summed = (out.last_hidden_state * mask).sum(1)
            counts = mask.sum(1).clamp(min=1e-9)
            all_emb.append((summed / counts).cpu().numpy())
    return np.concatenate(all_emb, axis=0) if all_emb else np.zeros((0, model.config.hidden_size))


def compute_phobert_features(ref_lines: list[str], hyp_lines: list[str], tokenizer, model, device: str = "cpu") -> dict:
    """Tính 5 đặc trưng PhoBERT từ embedding của dòng gốc và nghi vấn"""
    if not ref_lines or not hyp_lines:
        return {"mean_similarity": 0.0, "min_similarity": 0.0, "std_similarity": 0.0,
                "ref_to_hyp_mean": 0.0, "hyp_to_ref_mean": 0.0}

    ref_emb = _phobert_embed(ref_lines, tokenizer, model, device)
    hyp_emb = _phobert_embed(hyp_lines, tokenizer, model, device)

    ref_n = ref_emb / (np.linalg.norm(ref_emb, axis=1, keepdims=True) + 1e-9)
    hyp_n = hyp_emb / (np.linalg.norm(hyp_emb, axis=1, keepdims=True) + 1e-9)
    sim_matrix = ref_n @ hyp_n.T  # (n_ref, n_hyp)

    ref_to_hyp = sim_matrix.max(axis=1)  # mỗi dòng gốc -> dòng nghi vấn giống nhất
    hyp_to_ref = sim_matrix.max(axis=0)  # mỗi dòng nghi vấn -> dòng gốc giống nhất
    combined = np.concatenate([ref_to_hyp, hyp_to_ref])

    return {
        "mean_similarity": float(combined.mean()),
        "min_similarity": float(combined.min()),
        "std_similarity": float(combined.std()),
        "ref_to_hyp_mean": float(ref_to_hyp.mean()),
        "hyp_to_ref_mean": float(hyp_to_ref.mean()),
    }


def _normalize_bbox(bbox: list[float], width: int, height: int) -> list[int]:
    x0, y0, x1, y1 = bbox
    width, height = max(width, 1), max(height, 1)
    return [
        max(0, min(1000, int(1000 * x0 / width))),
        max(0, min(1000, int(1000 * y0 / height))),
        max(0, min(1000, int(1000 * x1 / width))),
        max(0, min(1000, int(1000 * y1 / height))),
    ]


def _layoutlmv3_embed(image: Image.Image, lines: list[dict], processor, model, device: str = "cpu") -> np.ndarray:
    """Tính embedding LayoutLMv3 từ ảnh trang và bounding box của dòng"""
    import torch
    words = [l["text"] if l["text"].strip() else "." for l in lines]
    boxes = [_normalize_bbox(l["bbox"], image.width, image.height) for l in lines]
    encoding = processor(image.convert("RGB"), words, boxes=boxes, return_tensors="pt",
                         truncation=True, padding="max_length")
    encoding = {k: v.to(device) for k, v in encoding.items()}
    with torch.no_grad():
        out = model(**encoding)
    return out.last_hidden_state.mean(dim=1).squeeze(0).cpu().numpy()


def compute_layoutlmv3_similarity(orig_image: Image.Image, orig_lines: list[dict],
                                   cand_image: Image.Image, cand_lines: list[dict],
                                   processor, model, device: str = "cpu") -> float:
    """Tính cosine similarity giữa embedding LayoutLMv3 của 2 ảnh trang"""
    if not orig_lines or not cand_lines:
        return 0.0
    orig_emb = _layoutlmv3_embed(orig_image, orig_lines, processor, model, device)
    cand_emb = _layoutlmv3_embed(cand_image, cand_lines, processor, model, device)
    cos = float(np.dot(orig_emb, cand_emb) / (np.linalg.norm(orig_emb) * np.linalg.norm(cand_emb) + 1e-9))
    return cos


def extract_hybrid_features(orig_pages_img: list[dict], orig_ocr: list[list[dict]],
                             cand_pages_img: list[dict], cand_ocr: list[list[dict]],
                             device: str = "cpu") -> dict:
    """
    Tính đủ 8 đặc trưng cho 1 cặp tài liệu (gốc + nghi vấn) đã OCR sẵn.
    
    8 đặc trưng:
    1. CER - Character Error Rate
    2. WER - Word Error Rate
    3. mean_similarity - Trung bình cosine similarity PhoBERT
    4. min_similarity - Minimum cosine similarity PhoBERT
    5. std_similarity - Độ lệch chuẩn cosine similarity PhoBERT
    6. ref_to_hyp_mean - Trung bình độ tương tự dòng gốc đến dòng nghi vấn giống nhất
    7. hyp_to_ref_mean - Trung bình độ tương tự dòng nghi vấn đến dòng gốc giống nhất
    8. layoutlmv3_cosine_similarity - Cosine similarity LayoutLMv3 giữa 2 ảnh trang
    """
    from transformers import AutoTokenizer, AutoModel, LayoutLMv3Processor, LayoutLMv3Model

    orig_text = "\n".join(l["text"] for page in orig_ocr for l in page)
    cand_text = "\n".join(l["text"] for page in cand_ocr for l in page)
    cer, wer = compute_document_cer_wer(orig_text, cand_text)

    orig_lines_all = [l["text"] for page in orig_ocr for l in page]
    cand_lines_all = [l["text"] for page in cand_ocr for l in page]

    print("  Load PhoBERT...")
    pho_tok = AutoTokenizer.from_pretrained("vinai/phobert-base-v2")
    pho_model = AutoModel.from_pretrained("vinai/phobert-base-v2").to(device)
    pho_feats = compute_phobert_features(orig_lines_all, cand_lines_all, pho_tok, pho_model, device)

    print("  Load LayoutLMv3...")
    lay_processor = LayoutLMv3Processor.from_pretrained("microsoft/layoutlmv3-base", apply_ocr=False)
    lay_model = LayoutLMv3Model.from_pretrained("microsoft/layoutlmv3-base").to(device)

    n_pages = min(len(orig_pages_img), len(cand_pages_img))
    layout_sims = []
    for p in range(n_pages):
        if not orig_ocr[p] or not cand_ocr[p]:
            continue
        try:
            sim = compute_layoutlmv3_similarity(
                orig_pages_img[p]["image"], orig_ocr[p],
                cand_pages_img[p]["image"], cand_ocr[p],
                lay_processor, lay_model, device,
            )
            layout_sims.append(sim)
        except Exception as e:
            print(f"  [WARN] LayoutLMv3 error on page {p}: {e}")
    layout_sim = float(np.mean(layout_sims)) if layout_sims else 0.0

    return {
        "cer": cer, "wer": wer,
        **pho_feats,
        "layoutlmv3_cosine_similarity": layout_sim,
    }


def load_hybrid_model(model_path: str, encoder_path: str):
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(encoder_path, "rb") as f:
        le = pickle.load(f)
    return model, le


def predict_with_hybrid_model(features: dict, model, le) -> dict:
    X = np.array([[features[c] for c in FEATURE_ORDER]])
    pred = model.predict(X)[0]
    proba = model.predict_proba(X)[0]
    label = le.inverse_transform([pred])[0]
    return {
        "label": label,
        "is_tampered": label != "1.original",
        "confidence": float(proba[pred]),
        "all_probs": {le.inverse_transform([i])[0]: float(p) for i, p in enumerate(proba)},
    }


# ============================
# 6. Xuất báo cáo HTML
# ============================
def generate_html(report: dict, out_dir: Path) -> str:
    lines_html = ""
    for page in report["pages"]:
        img_file = f"page_{page['page_index']:03d}_highlighted.png"
        lines_html += f"<h2>Trang {page['page_index'] + 1}</h2>"
        if page["tampered_lines"]:
            lines_html += f"<p>Tampered lines on this page: <strong>{len(page['tampered_lines'])}</strong></p>"
            lines_html += "<table border='1' cellpadding='6' style='border-collapse:collapse;width:100%'>"
            lines_html += "<tr><th>Line</th><th>Type</th><th>CER</th><th>Original Text</th><th>Candidate Text</th></tr>"
            for tl in page["tampered_lines"]:
                bg = "#f8d7da" if tl['type'] == "inserted" else "#fff3cd" if tl['type'] == "modified" else "#ffeeba"
                lines_html += (f"<tr style='background:{bg}'>"
                    f"<td>{tl.get('line_idx','?')}</td>"
                    f"<td><b>{tl['type'].upper()}</b></td>"
                    f"<td>{tl['cer']:.2%}</td>"
                    f"<td>{tl.get('orig_text','—')}</td>"
                    f"<td>{tl.get('cand_text','—')}</td></tr>")
            lines_html += "</table>"
            lines_html += f"<img src='{img_file}' style='width:100%;border:1px solid #ccc;margin-top:8px'/>"
        else:
            lines_html += "<p style='color:green'>&#10004; No tampering detected on this page.</p>"

    verdict = report["verdict"]
    verdict_color = "#d4edda" if verdict == "AUTHENTIC" else "#f8d7da"

    # Model prediction section
    model_html = ""
    if "model_prediction" in report:
        mp = report["model_prediction"]
        model_html = f"""
<div style='padding:16px;background:#e7f3fe;border:1px solid #b6d4fe;border-radius:8px;margin-bottom:20px'>
  <h2 style='margin:0;color:#084298'>Model Prediction (Hybrid Fusion - 5 class)</h2>
  <p><strong>Predicted Label:</strong> {mp['label']}<br>
     <strong>Is Tampered:</strong> {'Yes' if mp['is_tampered'] else 'No'}<br>
     <strong>Confidence:</strong> {mp['confidence']:.2%}</p>
  <p><strong>All Probabilities:</strong></p>
  <ul>"""
        for cls_name, prob in mp.get('all_probs', {}).items():
            model_html += f"<li>{cls_name}: {prob:.2%}</li>"
        model_html += "</ul></div>"

    # Features section
    features_html = ""
    if "hybrid_features" in report:
        features_html = "<div style='padding:16px;background:#f0f0f0;border-radius:8px;margin-bottom:20px'>"
        features_html += "<h2 style='margin:0'>Hybrid Features (8 features)</h2><ul>"
        for k, v in report["hybrid_features"].items():
            features_html += f"<li><strong>{k}:</strong> {v:.6f}</li>"
        features_html += "</ul></div>"

    html = f"""<!DOCTYPE html>
<html lang='vi'><head><meta charset='UTF-8'>
<title>Tampering Detection Report</title>
<style>body{{font-family:Arial,sans-serif;padding:20px;max-width:1200px;margin:auto}}
h1{{color:#1F4E79}}h2{{color:#2e75b6;margin-top:30px}}
table{{font-size:13px}}
</style></head><body>
<h1>&#128269; PDF Tampering Detection Report</h1>
<div style='padding:16px;background:{verdict_color};border-radius:8px;margin-bottom:20px'>
  <h2 style='margin:0'>Verdict: {verdict}</h2>
  <p>Original PDF: <code>{report['original_pdf']}</code><br>
     Candidate PDF: <code>{report['candidate_pdf']}</code><br>
     Total tampered lines: <strong>{report['total_tampered_lines']}</strong> / {report['total_lines']} lines</p>
</div>
{model_html}
{features_html}
{lines_html}
</body></html>"""
    return html

# ============================
# 7. Main
# ============================
def main():
    parser = argparse.ArgumentParser(description="Part 8 - Explainer: find tampered lines and highlight on images")
    parser.add_argument("--original", required=True, help="Path to original PDF")
    parser.add_argument("--candidate", required=True, help="Path to candidate (suspicious) PDF")
    parser.add_argument("--out-dir", default="tamper_output", help="Output directory")
    parser.add_argument("--cer-threshold", type=float, default=0.2,
                        help="Lines with CER > this threshold are considered tampered")
    parser.add_argument("--dpi", type=int, default=150, help="DPI for rendering")
    parser.add_argument("--ocr-backend", choices=["pymupdf", "paddleocr"], default="pymupdf",
                        help="Text extraction backend: pymupdf (fast, embedded text) or paddleocr (slow, OCR)")
    parser.add_argument("--model", default=None, help="Path to hybrid_model.pkl (optional)")
    parser.add_argument("--encoder", default=None, help="Path to label_encoder.pkl (optional)")
    parser.add_argument("--device", default="cpu", help="Device for models (cpu or cuda)")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True)

    # Auto-detect model path if not specified
    script_dir = Path(__file__).parent
    model_path = Path(args.model) if args.model else script_dir / "hybrid_model.pkl"
    encoder_path = Path(args.encoder) if args.encoder else script_dir / "label_encoder.pkl"

    print("Rendering PDF pages...")
    orig_pages = render_pages(Path(args.original), args.dpi)
    cand_pages = render_pages(Path(args.candidate), args.dpi)
    print(f"  Original: {len(orig_pages)} pages, Candidate: {len(cand_pages)} pages")

    print(f"Extracting text (backend: {args.ocr_backend})...")
    if args.ocr_backend == "pymupdf":
        orig_ocr = extract_text_pymupdf(Path(args.original), args.dpi)
        cand_ocr = extract_text_pymupdf(Path(args.candidate), args.dpi)
    else:
        orig_ocr = ocr_pages(Path(args.original), args.dpi)
        cand_ocr = ocr_pages(Path(args.candidate), args.dpi)

    orig_total_lines = sum(len(p) for p in orig_ocr)
    cand_total_lines = sum(len(p) for p in cand_ocr)
    print(f"  Original: {orig_total_lines} lines, Candidate: {cand_total_lines} lines")

    # ---- Part 7: Model Prediction ----
    model_prediction = None
    hybrid_features = None
    if model_path.exists() and encoder_path.exists():
        print(f"\nLoading hybrid model from {model_path}...")
        hybrid_model, label_encoder = load_hybrid_model(str(model_path), str(encoder_path))

        print("Computing 8 hybrid features (PhoBERT + LayoutLMv3)...")
        hybrid_features = extract_hybrid_features(
            orig_pages, orig_ocr,
            cand_pages, cand_ocr,
            device=args.device
        )

        print("\n=== HYBRID FEATURES ===")
        for k, v in hybrid_features.items():
            print(f"  {k:35s} = {v:.6f}")

        model_prediction = predict_with_hybrid_model(hybrid_features, hybrid_model, label_encoder)
        print(f"\n=== MODEL PREDICTION ===")
        print(f"  Predicted Label : {model_prediction['label']}")
        print(f"  Is Tampered     : {model_prediction['is_tampered']}")
        print(f"  Confidence      : {model_prediction['confidence']:.2%}")
        print(f"  All probabilities:")
        for cls_name, prob in model_prediction['all_probs'].items():
            print(f"    {cls_name:20s}: {prob:7.2%}")
    else:
        print(f"\n[INFO] Model not found ({model_path}). Skipping model prediction.")
        print(f"       Only line-by-line comparison will be done.")

    # ---- Part 8: Document-level line alignment and highlighting ----
    print(f"\nComparing lines (CER threshold = {args.cer_threshold})...")
    print(f"  Using DOCUMENT-LEVEL alignment (cross-page) to reduce false positives...")

    report = {
        "original_pdf": args.original,
        "candidate_pdf": args.candidate,
        "total_lines": 0,
        "total_tampered_lines": 0,
        "verdict": "AUTHENTIC",
        "pages": [],
    }
    if model_prediction:
        report["model_prediction"] = model_prediction
    if hybrid_features:
        report["hybrid_features"] = hybrid_features

    # Flatten all lines across pages, keeping page_idx for mapping back
    all_orig_lines = []
    for page_idx, page_lines in enumerate(orig_ocr):
        for line in page_lines:
            all_orig_lines.append({**line, "page_idx": page_idx})

    all_cand_lines = []
    for page_idx, page_lines in enumerate(cand_ocr):
        for line in page_lines:
            all_cand_lines.append({**line, "page_idx": page_idx})

    # Document-level DP alignment (no sorting by Y, already in reading order)
    print(f"  Aligning {len(all_orig_lines)} orig lines vs {len(all_cand_lines)} cand lines...")
    doc_pairs = match_lines(all_orig_lines, all_cand_lines, sort_by_y=False)

    # Classify tampered pairs and group by candidate page
    n_pages = max(len(orig_ocr), len(cand_ocr))
    page_tampered_pairs = {i: [] for i in range(n_pages)}
    page_tampered_info = {i: [] for i in range(n_pages)}

    total_matched = 0
    for pair in doc_pairs:
        total_matched += 1
        t = pair["type"]
        if t == "match" and pair["cer"] <= args.cer_threshold:
            continue  # identical or near-identical → skip

        # Classify modified
        if t == "match" and pair["cer"] > args.cer_threshold:
            t = "modified"
        pair["type"] = t

        # Determine which page to highlight on
        cand_line = pair.get("cand_line")
        orig_line = pair.get("orig_line")
        if cand_line:
            pg = cand_line.get("page_idx", 0)
        elif orig_line:
            pg = orig_line.get("page_idx", 0)
        else:
            pg = 0

        if pg < n_pages:
            page_tampered_pairs[pg].append(pair)
            page_tampered_info[pg].append({
                "line_idx": (cand_line or orig_line).get("line_idx"),
                "type": t,
                "cer": round(pair["cer"], 4),
                "orig_text": orig_line["text"] if orig_line else "",
                "cand_text": cand_line["text"] if cand_line else "",
                "bbox": (cand_line or orig_line)["bbox"],
            })

    # Generate highlighted images and build report pages
    for page_idx in range(n_pages):
        tampered_pairs = page_tampered_pairs[page_idx]
        tampered_info = page_tampered_info[page_idx]

        # Highlight on image
        if page_idx < len(cand_pages):
            cand_img = cand_pages[page_idx]["image"].copy()
        else:
            cand_img = Image.new("RGB", (800, 100), "white")
        if tampered_pairs:
            cand_img = highlight_page(cand_img, tampered_pairs)
        img_file = out_dir / f"page_{page_idx:03d}_highlighted.png"
        cand_img.save(img_file)

        page_line_count = len(orig_ocr[page_idx]) if page_idx < len(orig_ocr) else 0
        page_line_count = max(page_line_count, len(cand_ocr[page_idx]) if page_idx < len(cand_ocr) else 0)

        report["total_lines"] += page_line_count
        report["total_tampered_lines"] += len(tampered_info)
        report["pages"].append({
            "page_index": page_idx,
            "tampered_lines": tampered_info,
            "image_file": str(img_file),
        })

    report["verdict"] = "AUTHENTIC" if report["total_tampered_lines"] == 0 else "TAMPERED"

    # Save JSON report
    json_path = out_dir / "tampered_report.json"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    # Save HTML report
    html_content = generate_html(report, out_dir)
    html_path = out_dir / "tampered_report.html"
    html_path.write_text(html_content, encoding="utf-8")

    print(f"\n{'='*60}")
    print(f"TAMPERING DETECTION RESULT")
    print(f"{'='*60}")
    print(f"Verdict                : {report['verdict']}")
    print(f"Total tampered lines   : {report['total_tampered_lines']}/{report['total_lines']} lines")
    if report['total_lines'] > 0:
        pct = 100.0 * report['total_tampered_lines'] / report['total_lines']
        print(f"Tampering percentage   : {pct:.1f}%")
    if model_prediction:
        print(f"Model prediction       : {model_prediction['label']} ({model_prediction['confidence']:.2%})")
    print(f"HTML Report            : {html_path}")
    print(f"JSON Report            : {json_path}")
    print(f"Highlighted images     : {out_dir}/page_*.png")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()