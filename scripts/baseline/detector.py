#!/usr/bin/env python3
"""
Phần 7 – Tampering Detector
Input : 1 PDF nghi vấn + 1 PDF gốc tham chiếu
Output: nhãn (Authentic / Tampered), loại tấn công dự đoán, confidence

Sử dụng model hybrid fusion đã train (RandomForest) với 8 đặc trưng:
  - CER, WER (Levenshtein)
  - 5 đặc trưng PhoBERT (mean/min/std similarity, ref_to_hyp, hyp_to_ref)
  - LayoutLMv3 cosine similarity

Hỗ trợ 2 backend trích xuất text:
  - pymupdf (mặc định) : nhanh, dùng embedded text từ PDF (born-digital)
  - paddleocr           : chậm hơn, dùng OCR (phù hợp PDF scan)
"""
import argparse
import json
import pickle
from pathlib import Path
import numpy as np
import sys

# Import từ explainer.py để tái dùng các hàm
sys.path.insert(0, str(Path(__file__).parent))
from explainer import (
    render_pages, ocr_pages, extract_text_pymupdf,
    extract_hybrid_features, FEATURE_ORDER
)

def main():
    parser = argparse.ArgumentParser(description="Part 7 - Tampering Detector")
    parser.add_argument("--original", required=True, help="Path to original reference PDF")
    parser.add_argument("--candidate", required=True, help="Path to candidate (suspicious) PDF")
    parser.add_argument("--model", default=None, help="Path to hybrid_model.pkl")
    parser.add_argument("--encoder", default=None, help="Path to label_encoder.pkl")
    parser.add_argument("--device", default="cpu", help="cuda or cpu")
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--ocr-backend", choices=["pymupdf", "paddleocr"], default="pymupdf",
                        help="Text extraction backend: pymupdf (fast) or paddleocr (slow)")
    args = parser.parse_args()

    original_path = Path(args.original)
    candidate_path = Path(args.candidate)

    # Auto-detect model path if not specified
    script_dir = Path(__file__).parent
    model_path = Path(args.model) if args.model else script_dir / "hybrid_model.pkl"
    encoder_path = Path(args.encoder) if args.encoder else script_dir / "label_encoder.pkl"

    if not original_path.exists():
        print(f"[ERROR] File not found: {original_path}", file=sys.stderr)
        return 1
    if not candidate_path.exists():
        print(f"[ERROR] File not found: {candidate_path}", file=sys.stderr)
        return 1

    # Load model
    if not model_path.exists() or not encoder_path.exists():
        print(f"[ERROR] Model files not found: {model_path}, {encoder_path}")
        print(f"        Please train the model first (see Tuan5/hybrid_fusion/train_hybrid_fusion.py)")
        return 1

    print(f"Loading model from {model_path}...")
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(encoder_path, "rb") as f:
        le = pickle.load(f)

    print(f"\nRendering original PDF...")
    orig_pages = render_pages(original_path, args.dpi)
    print(f"  -> {len(orig_pages)} pages")

    print(f"Rendering candidate PDF...")
    cand_pages = render_pages(candidate_path, args.dpi)
    print(f"  -> {len(cand_pages)} pages")

    print(f"\nExtracting text (backend: {args.ocr_backend})...")
    if args.ocr_backend == "pymupdf":
        orig_ocr = extract_text_pymupdf(original_path, args.dpi)
        cand_ocr = extract_text_pymupdf(candidate_path, args.dpi)
    else:
        orig_ocr = ocr_pages(original_path, args.dpi)
        cand_ocr = ocr_pages(candidate_path, args.dpi)

    orig_lines = sum(len(p) for p in orig_ocr)
    cand_lines = sum(len(p) for p in cand_ocr)
    print(f"  Original: {orig_lines} lines, Candidate: {cand_lines} lines")

    print(f"\nComputing hybrid features...")
    features = extract_hybrid_features(
        orig_pages, orig_ocr,
        cand_pages, cand_ocr,
        device=args.device
    )

    print(f"\n=== FEATURES ===")
    for k, v in features.items():
        print(f"  {k:35s} = {v:.6f}")

    # Predict
    X = np.array([[features[c] for c in FEATURE_ORDER]])
    pred = model.predict(X)[0]
    proba = model.predict_proba(X)[0]
    label = le.inverse_transform([pred])[0]
    confidence = proba[pred]

    print(f"\n{'='*60}")
    print(f"DETECTION RESULT")
    print(f"{'='*60}")
    print(f"Predicted label    : {label}")
    is_authentic = label == "1.original"
    print(f"Status             : {'AUTHENTIC (Original)' if is_authentic else 'TAMPERED (Modified)'}")
    print(f"Confidence         : {confidence:.2%}")
    print(f"\nProbability details:")
    for i, prob in enumerate(proba):
        cat = le.inverse_transform([i])[0]
        print(f"  {cat:20s}: {prob:7.2%}")
    print(f"{'='*60}")

    return 0

if __name__ == "__main__":
    sys.exit(main())