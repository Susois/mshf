#!/usr/bin/env python3
"""
Batch Pipeline - Hợp nhất Part 7 (Detector) + Part 8 (Explainer)
==================================================================
Chạy toàn bộ PDF trong 1.pdfs/{2.insert,3.delete,4.modify,5.layout}/*.pdf,
so sánh THÔ (raw) với PDF gốc thật trong 1.pdfs/1.original/:
  - TEXT   : trích xuất trực tiếp từ PDF gốc bằng PyMuPDF (KHÔNG dùng ground truth .txt)
  - LAYOUT : ảnh render trực tiếp từ PDF gốc, so sánh LayoutLMv3 với ảnh candidate

Tái sử dụng TOÀN BỘ các hàm gốc trong explainer.py (render_pages,
extract_text_pymupdf, compute_document_cer_wer, compute_phobert_features,
compute_layoutlmv3_similarity, match_lines, highlight_page, predict_with_hybrid_model,
generate_html...) - đây CHÍNH LÀ các hàm mà detector.py cũng import và dùng,
nên chạy batch này tương đương với việc chạy detector.py cho từng cặp tài liệu,
cộng thêm phần highlight & report của explainer.py.

Output structure:
  output/<attack_type>/<pdf_name>/
      report.json
      report.html
      page_000_highlighted.png, page_001_highlighted.png, ...
"""
import argparse
import json
import pickle
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
from PIL import Image

# Import lại toàn bộ hàm gốc từ explainer.py (đặt cùng thư mục)
sys.path.insert(0, str(Path(__file__).parent))
from explainer import (
    render_pages, extract_text_pymupdf,
    compute_document_cer_wer, compute_phobert_features,
    compute_layoutlmv3_similarity,
    match_lines, highlight_page, generate_html,
    load_hybrid_model, predict_with_hybrid_model,
    FEATURE_ORDER, CER_THRESHOLD,
)

# Tên thư mục tấn công = TÊN NHÃN THẬT mà model đã học (khớp label_encoder classes)
ATTACK_TYPES = ["2.insert", "3.delete", "4.modify", "5.layout"]


# ============================================================
# Tính 8 đặc trưng hybrid - TOÀN BỘ dùng raw scan (PyMuPDF) từ
# PDF gốc thật vs PDF candidate, không phụ thuộc ground truth
# ============================================================
def extract_features_raw(
    orig_pages_img: list, orig_ocr: list,
    cand_pages_img: list, cand_ocr: list,
    pho_tok, pho_model, lay_processor, lay_model,
    device: str = "cpu",
) -> dict:
    orig_text = "\n".join(l["text"] for page in orig_ocr for l in page)
    cand_text = "\n".join(l["text"] for page in cand_ocr for l in page)
    cer, wer = compute_document_cer_wer(orig_text, cand_text)

    orig_lines_all = [l["text"] for page in orig_ocr for l in page]
    cand_lines_all = [l["text"] for page in cand_ocr for l in page]
    pho_feats = compute_phobert_features(orig_lines_all, cand_lines_all, pho_tok, pho_model, device)

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
            print(f"    [WARN] LayoutLMv3 lỗi ở trang {p}: {e}")
    layout_sim = float(np.mean(layout_sims)) if layout_sims else 0.0

    return {
        "cer": cer, "wer": wer,
        **pho_feats,
        "layoutlmv3_cosine_similarity": layout_sim,
    }


# ============================================================
# Xử lý 1 cặp (PDF gốc, candidate PDF) -> report + ảnh highlight
# ============================================================
def process_single_pdf(
    pdf_stem: str, attack_type: str, cand_pdf_path: Path,
    orig_pages_img: list, orig_ocr: list,
    model, label_encoder,
    pho_tok, pho_model, lay_processor, lay_model,
    out_base: Path, device: str = "cpu", dpi: int = 150,
    cer_threshold: float = CER_THRESHOLD,
) -> dict:
    try:
        pdf_out_dir = out_base / attack_type / pdf_stem
        pdf_out_dir.mkdir(parents=True, exist_ok=True)

        cand_pages = render_pages(cand_pdf_path, dpi)
        cand_ocr = extract_text_pymupdf(cand_pdf_path, dpi)

        # ---- 8 đặc trưng hybrid (raw scan, không dùng ground truth) ----
        features = extract_features_raw(
            orig_pages_img, orig_ocr,
            cand_pages, cand_ocr,
            pho_tok, pho_model, lay_processor, lay_model,
            device=device,
        )

        # ---- Dự đoán nhãn (tương đương detector.py) ----
        model_prediction = predict_with_hybrid_model(features, model, label_encoder)

        # ---- Căn dòng để highlight (document-level, không sort theo y) ----
        all_orig_lines = []
        for page_idx, page_lines in enumerate(orig_ocr):
            for line in page_lines:
                all_orig_lines.append({**line, "page_idx": page_idx})

        all_cand_lines = []
        for page_idx, page_lines in enumerate(cand_ocr):
            for line in page_lines:
                all_cand_lines.append({**line, "page_idx": page_idx})

        doc_pairs = match_lines(all_orig_lines, all_cand_lines, sort_by_y=False)

        n_pages = len(cand_pages)
        page_tampered_pairs = {i: [] for i in range(n_pages)}
        page_tampered_info = {i: [] for i in range(n_pages)}
        total_tampered = 0

        for pair in doc_pairs:
            t = pair["type"]
            if t == "match" and pair["cer"] <= cer_threshold:
                continue
            if t == "match" and pair["cer"] > cer_threshold:
                t = "modified"
            pair["type"] = t

            cand_line = pair.get("cand_line")
            orig_line = pair.get("orig_line")

            # Dòng "deleted" (có trong PDF gốc, không còn trong candidate) vẫn được
            # vẽ highlight, dùng bbox từ trang gốc (vị trí xấp xỉ trên ảnh candidate)
            if cand_line:
                pg = cand_line.get("page_idx", 0)
            elif orig_line:
                pg = min(orig_line.get("page_idx", 0), n_pages - 1) if n_pages else 0
            else:
                pg = 0

            total_tampered += 1
            info = {
                "line_idx": (cand_line or orig_line).get("line_idx"),
                "type": t,
                "cer": round(pair["cer"], 4),
                "orig_text": orig_line["text"] if orig_line else "",
                "cand_text": cand_line["text"] if cand_line else "",
            }
            page_tampered_info[pg].append(info)
            page_tampered_pairs[pg].append(pair)

        report = {
            "original_pdf": "1.original (raw scan PyMuPDF)",
            "candidate_pdf": str(cand_pdf_path),
            "total_lines": sum(len(p) for p in cand_ocr),
            "total_tampered_lines": total_tampered,
            "verdict": "AUTHENTIC" if total_tampered == 0 else "TAMPERED",
            "model_prediction": model_prediction,
            "hybrid_features": features,
            "pages": [],
        }

        for page_idx in range(n_pages):
            img = cand_pages[page_idx]["image"].copy()
            if page_tampered_pairs[page_idx]:
                img = highlight_page(img, page_tampered_pairs[page_idx])
            img_path = pdf_out_dir / f"page_{page_idx:03d}_highlighted.png"
            img.save(img_path)

            report["pages"].append({
                "page_index": page_idx,
                "tampered_lines": page_tampered_info[page_idx],
                "image_file": str(img_path),
            })

        (pdf_out_dir / "report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (pdf_out_dir / "report.html").write_text(
            generate_html(report, pdf_out_dir), encoding="utf-8"
        )

        return {
            "status": "success",
            "pdf": pdf_stem,
            "attack_type": attack_type,
            "verdict": report["verdict"],
            "label": model_prediction["label"],
            "confidence": model_prediction["confidence"],
            "total_tampered": total_tampered,
            "output_dir": str(pdf_out_dir),
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"status": "error", "pdf": pdf_stem, "attack_type": attack_type, "error": str(e)}


def main():
    parser = argparse.ArgumentParser(description="Batch Detector + Explainer (raw scan, không dùng ground truth)")
    parser.add_argument("--pdfs-dir", default="1.pdfs")
    parser.add_argument("--model", default=None)
    parser.add_argument("--encoder", default=None)
    parser.add_argument("--output", default="output")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--cer-threshold", type=float, default=CER_THRESHOLD)
    args = parser.parse_args()

    pdfs_dir = Path(args.pdfs_dir)
    out_base = Path(args.output)
    out_base.mkdir(exist_ok=True)

    script_dir = Path(__file__).parent
    model_path = Path(args.model) if args.model else script_dir / "hybrid_model.pkl"
    encoder_path = Path(args.encoder) if args.encoder else script_dir / "label_encoder.pkl"

    print(f"Loading model từ {model_path}...")
    model, label_encoder = load_hybrid_model(str(model_path), str(encoder_path))

    print("Pre-load PhoBERT & LayoutLMv3 (1 lần duy nhất cho toàn batch)...")
    from transformers import AutoTokenizer, AutoModel, LayoutLMv3Processor, LayoutLMv3Model
    pho_tok = AutoTokenizer.from_pretrained("vinai/phobert-base-v2")
    pho_model = AutoModel.from_pretrained("vinai/phobert-base-v2").to(args.device)
    pho_model.eval()
    lay_processor = LayoutLMv3Processor.from_pretrained("microsoft/layoutlmv3-base", apply_ocr=False)
    lay_model = LayoutLMv3Model.from_pretrained("microsoft/layoutlmv3-base").to(args.device)
    lay_model.eval()
    print("Models đã sẵn sàng. Bắt đầu batch...\n")

    original_dir = pdfs_dir / "1.original"
    orig_pdfs = sorted(original_dir.glob("*.pdf"))
    print(f"Tìm thấy {len(orig_pdfs)} PDF gốc trong {original_dir}\n")
    if not orig_pdfs:
        print(f"[ERROR] Không có PDF nào trong {original_dir}. Hãy thêm PDF gốc vào đây trước.")
        return 1

    all_results = []
    summary = defaultdict(lambda: {"total": 0, "correct": 0, "tampered": 0, "confidence": 0.0})

    for orig_pdf in orig_pdfs:
        pdf_stem = orig_pdf.stem

        # Render + extract PDF GỐC THẬT (chỉ 1 lần cho mỗi document, dùng lại cho cả 4 loại tấn công)
        orig_pages_img = render_pages(orig_pdf, args.dpi)
        orig_ocr = extract_text_pymupdf(orig_pdf, args.dpi)

        print(f"{pdf_stem:45s} ", end="", flush=True)

        for attack_type in ATTACK_TYPES:
            attack_name = attack_type.split(".")[-1]
            cand_pdf = pdfs_dir / attack_type / orig_pdf.name

            if not cand_pdf.exists():
                print("⊘", end="", flush=True)
                continue

            result = process_single_pdf(
                pdf_stem, attack_type, cand_pdf,
                orig_pages_img, orig_ocr,
                model, label_encoder,
                pho_tok, pho_model, lay_processor, lay_model,
                out_base, device=args.device, dpi=args.dpi,
                cer_threshold=args.cer_threshold,
            )

            if result["status"] == "success":
                # Tên thư mục tấn công CHÍNH LÀ nhãn thật (đã khớp label_encoder)
                expected_label = attack_type
                is_correct = result["label"] == expected_label
                icon = "✅" if is_correct else "❌"
                print(icon, end="", flush=True)

                summary[attack_name]["total"] += 1
                summary[attack_name]["correct"] += int(is_correct)
                summary[attack_name]["tampered"] += int(result["verdict"] == "TAMPERED")
                summary[attack_name]["confidence"] += result["confidence"]
                all_results.append(result)
            else:
                print("E", end="", flush=True)
                all_results.append(result)

        print()

    print(f"\n{'='*70}")
    print("BATCH PROCESSING SUMMARY")
    print(f"{'='*70}\n")

    total_correct, total_count = 0, 0
    for attack_type in ATTACK_TYPES:
        attack_name = attack_type.split(".")[-1]
        s = summary[attack_name]
        if s["total"] == 0:
            continue
        acc = s["correct"] / s["total"] * 100
        detect = s["tampered"] / s["total"] * 100
        conf = s["confidence"] / s["total"] * 100
        total_correct += s["correct"]
        total_count += s["total"]
        print(f"{attack_name.upper():12s} : {s['total']:2d} tests | "
              f"Accuracy: {acc:5.1f}% | Detected: {detect:5.1f}% | Confidence: {conf:5.1f}%")

    if total_count > 0:
        print(f"\n{'OVERALL':12s} : {total_count:2d} tests | Accuracy: {total_correct/total_count*100:5.1f}%")

    results_json = out_base / "all_results.json"
    results_json.write_text(json.dumps(all_results, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n✅ Kết quả đã lưu vào {out_base}/")
    print(f"   Cấu trúc: {out_base}/<attack_type>/<pdf_name>/")
    print("   File: report.json | report.html | page_*_highlighted.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())