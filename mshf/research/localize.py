"""Định vị dòng bị can thiệp + line-level P/R/F1 trên VEDTD.

Inference chỉ dùng original OCR vs candidate OCR. Ground-truth sạch chỉ dùng để đánh giá.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import config  # noqa: E402
from mshf.core import io_utils  # noqa: E402
from mshf.core.line_align import match_lines  # noqa: E402
from mshf.core.localization_gt import ground_truth_spans  # noqa: E402
from mshf.core.semantic_risk import semantic_risk  # noqa: E402


def predict_events(doc_id: str, category: str) -> list[dict]:
    ref = io_utils.read_text_lines(io_utils.ocr_text_path(config.ORIGINAL_CAT, doc_id))
    cand = io_utils.read_text_lines(io_utils.ocr_text_path(category, doc_id))
    ref_d = [{"text": t, "bbox": [0, i, 1, i + 1], "idx": i} for i, t in enumerate(ref)]
    cand_d = [{"text": t, "bbox": [0, i, 1, i + 1], "idx": i} for i, t in enumerate(cand)]
    pairs = match_lines(ref_d, cand_d)
    events = []
    for pos, p in enumerate(pairs):
        typ = p["type"]
        if typ == "match":
            risk = semantic_risk(p["orig_line"]["text"], p["cand_line"]["text"])
            if p["cer"] > config.CER_MODIFIED_THRESHOLD or risk["flags"]:
                events.append({"type": "modified", "candidate_line": p["cand_line"]["idx"], "cer": round(p["cer"], 4), "flags": risk["flags"], "reference_text": p["orig_line"]["text"], "candidate_text": p["cand_line"]["text"]})
        elif typ == "inserted":
            events.append({"type": "inserted", "candidate_line": p["cand_line"]["idx"], "cer": 1.0, "flags": [], "reference_text": "", "candidate_text": p["cand_line"]["text"]})
        elif typ == "deleted":
            anchor = next((q["cand_line"]["idx"] for q in pairs[pos + 1:] if q["cand_line"] is not None), None)
            events.append({"type": "deleted", "candidate_line": anchor, "cer": 1.0, "flags": [], "reference_text": p["orig_line"]["text"], "candidate_text": ""})
    return events


def main() -> None:
    ap = argparse.ArgumentParser(description="MSHF line localization evaluation")
    ap.add_argument("--out", type=Path, default=config.OUTPUT_DIR / "localization")
    ap.add_argument("--max-per-cat", type=int, default=0)
    args = ap.parse_args(); args.out.mkdir(parents=True, exist_ok=True)
    rows, details = [], []
    for cat in ["2.insert", "3.delete", "4.modify", "5.layout"]:
        docs = io_utils.discover_doc_ids()
        if args.max_per_cat: docs = docs[:args.max_per_cat]
        tp = fp = fn = 0
        for doc in docs:
            pred = predict_events(doc, cat)
            gt = ground_truth_spans(doc, cat)
            if cat == "2.insert":
                pred_set = {e["candidate_line"] for e in pred if e["type"] == "inserted"}
                gt_set = set(gt["inserted"])
            elif cat == "3.delete":
                pred_set = {e["candidate_line"] for e in pred if e["type"] == "deleted" and e["candidate_line"] is not None}
                gt_set = set(gt["deleted_after"])
            elif cat == "4.modify":
                pred_set = {e["candidate_line"] for e in pred if e["type"] == "modified"}
                gt_set = set(gt["modified"])
            else:
                pred_set, gt_set = set(), set()  # layout không đổi nội dung => không có line content GT
            d_tp = len(pred_set & gt_set); d_fp = len(pred_set - gt_set); d_fn = len(gt_set - pred_set)
            tp += d_tp; fp += d_fp; fn += d_fn
            details.append({"category": cat, "doc_id": doc, "tp": d_tp, "fp": d_fp, "fn": d_fn, "predicted_events": pred})
        precision = tp / max(tp + fp, 1); recall = tp / max(tp + fn, 1); f1 = 2 * precision * recall / max(precision + recall, 1e-12)
        rows.append({"category": cat, "documents": len(docs), "tp": tp, "fp": fp, "fn": fn, "precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4)})
        print(f"{cat}: P={precision:.4f} R={recall:.4f} F1={f1:.4f} (TP={tp},FP={fp},FN={fn})")
    with (args.out / "localization_metrics.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    (args.out / "localization_details.json").write_text(json.dumps(details, ensure_ascii=False, indent=2), encoding="utf-8")

if __name__ == "__main__": main()
