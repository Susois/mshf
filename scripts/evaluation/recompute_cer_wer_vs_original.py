# recompute_cer_wer_vs_original.py
from __future__ import annotations

import argparse
import csv
import re
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

try:
    from rapidfuzz.distance import Levenshtein
except Exception:
    Levenshtein = None

CATEGORIES = ["1.original", "2.insert", "3.delete", "4.modify", "5.layout"]


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8-sig", errors="ignore")


def normalize_text(text: str, *, case_sensitive: bool = False) -> str:
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\u00a0", " ")
    text = re.sub(r"\s+", " ", text).strip()
    if not case_sensitive:
        text = text.lower()
    return text


def python_edit_distance(a, b) -> int:
    a, b = list(a), list(b)
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            current.append(min(current[j - 1] + 1, previous[j] + 1, previous[j - 1] + (ca != cb)))
        previous = current
    return previous[-1]


def edit_distance(a, b) -> int:
    if Levenshtein is not None:
        return int(Levenshtein.distance(a, b))
    return python_edit_distance(a, b)


def compute_cer_wer(ref_text: str, hyp_text: str, *, case_sensitive: bool = False) -> dict[str, Any]:
    ref_norm = normalize_text(ref_text, case_sensitive=case_sensitive)
    hyp_norm = normalize_text(hyp_text, case_sensitive=case_sensitive)
    ref_words = ref_norm.split() if ref_norm else []
    hyp_words = hyp_norm.split() if hyp_norm else []

    char_edits = edit_distance(ref_norm, hyp_norm)
    word_edits = edit_distance(ref_words, hyp_words)
    ref_chars, hyp_chars = len(ref_norm), len(hyp_norm)
    ref_word_count, hyp_word_count = len(ref_words), len(hyp_words)

    cer = char_edits / ref_chars if ref_chars else (0.0 if not hyp_chars else 1.0)
    wer = word_edits / ref_word_count if ref_word_count else (0.0 if not hyp_word_count else 1.0)

    return {
        "cer": cer, "wer": wer, "char_edits": char_edits, "word_edits": word_edits,
        "ref_chars": ref_chars, "hyp_chars": hyp_chars,
        "ref_words": ref_word_count, "hyp_words": hyp_word_count,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Tính lại CER/WER: so OCR output của MỌI category với ground truth của 1.original."
    )
    parser.add_argument("--root", default=".")
    parser.add_argument("--gt-dir", default="2.ground_truth")
    parser.add_argument("--ocr-dir", default="3.ocr_output")
    parser.add_argument("--report", default="ocr_eval_report_vs_original.csv")
    parser.add_argument("--summary", default="ocr_eval_summary_vs_original.csv")
    parser.add_argument("--case-sensitive", action="store_true")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    gt_root = root / args.gt_dir
    ocr_root = root / args.ocr_dir
    report_path = root / args.report
    summary_path = root / args.summary

    original_gt_dir = gt_root / "1.original"
    if not original_gt_dir.exists():
        print(f"[ERROR] Không thấy: {original_gt_dir}", file=sys.stderr)
        return 1

    rows: list[dict[str, Any]] = []

    for category in CATEGORIES:
        ocr_cat_dir = ocr_root / category
        if not ocr_cat_dir.exists():
            print(f"[WARN] Bỏ qua, không thấy: {ocr_cat_dir}")
            continue

        txt_files = sorted(ocr_cat_dir.rglob("*.txt"))
        print(f"\n[{category}] {len(txt_files)} file OCR")

        for idx, ocr_txt_path in enumerate(txt_files, 1):
            rel = ocr_txt_path.relative_to(ocr_cat_dir)
            # LUÔN lấy ground truth từ 1.original, không phải từ category hiện tại
            gt_path = original_gt_dir / rel

            row: dict[str, Any] = {
                "category": category,
                "file": str(rel).replace("\\", "/"),
                "status": "",
                "error": "",
                "cer": "", "cer_percent": "", "wer": "", "wer_percent": "",
                "char_edits": "", "ref_chars": "", "hyp_chars": "",
                "word_edits": "", "ref_words": "", "hyp_words": "",
                "ground_truth_path": str(gt_path),
                "ocr_txt_path": str(ocr_txt_path),
            }

            try:
                if not gt_path.exists():
                    row["status"] = "missing_original_ground_truth"
                    print(f"  [{idx}/{len(txt_files)}] MISSING GT gốc: {rel}")
                    rows.append(row)
                    continue

                ref_text = read_text(gt_path)            # luôn là ground truth của 1.original
                hyp_text = read_text(ocr_txt_path)        # OCR output của category hiện tại

                metrics = compute_cer_wer(ref_text, hyp_text, case_sensitive=args.case_sensitive)
                row.update({
                    "cer": f'{metrics["cer"]:.8f}',
                    "cer_percent": f'{metrics["cer"] * 100:.4f}',
                    "wer": f'{metrics["wer"]:.8f}',
                    "wer_percent": f'{metrics["wer"] * 100:.4f}',
                    "char_edits": metrics["char_edits"],
                    "ref_chars": metrics["ref_chars"],
                    "hyp_chars": metrics["hyp_chars"],
                    "word_edits": metrics["word_edits"],
                    "ref_words": metrics["ref_words"],
                    "hyp_words": metrics["hyp_words"],
                    "status": "ok",
                })
                print(f"  [{idx}/{len(txt_files)}] OK: {rel}  CER={metrics['cer']*100:.2f}%")

            except Exception as e:
                row["status"] = "error"
                row["error"] = repr(e)
                print(f"  [ERROR] {rel}: {repr(e)}", file=sys.stderr)

            rows.append(row)

    fieldnames = list(rows[0].keys())
    with report_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    grouped: dict[str, list] = defaultdict(list)
    for r in rows:
        if r["status"] == "ok":
            grouped[r["category"]].append(r)

    summary_rows = []
    for category in CATEGORIES:
        group = grouped.get(category, [])
        if not group:
            continue
        macro_cer = mean(float(r["cer"]) for r in group)
        macro_wer = mean(float(r["wer"]) for r in group)
        summary_rows.append({
            "category": category, "num_docs": len(group),
            "macro_avg_cer_percent": f"{macro_cer*100:.4f}",
            "macro_avg_wer_percent": f"{macro_wer*100:.4f}",
        })

    with summary_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["category", "num_docs", "macro_avg_cer_percent", "macro_avg_wer_percent"])
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"\nĐã ghi: {report_path}")
    print(f"Đã ghi: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())