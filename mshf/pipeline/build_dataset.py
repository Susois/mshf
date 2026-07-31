"""Hợp nhất 3 nhánh đặc trưng theo doc_id -> outputs/enhanced_dataset.csv.

Nhánh A (GỐC, giữ nguyên): 8 feature doc-level nạp thẳng từ hybrid_fusion_dataset.csv
                           của đề tài gốc (không tính lại, không sửa).
Nhánh B (MỚI): đặc trưng mức DÒNG (line_features) từ OCR text.
Nhánh C (MỚI): đặc trưng HÌNH HỌC (geometric_features) từ layout JSON.

Mỗi dòng ứng với một cặp (1.original, category) của cùng doc_id, kèm:
  - label            : nhãn multi-class (original/insert/delete/modify/layout)
  - is_tampered      : 0 cho original, 1 cho 4 nhóm còn lại
  - source_document_id: doc_id  => key cho GroupKFold (5 biến thể cùng doc cùng group)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import config  # noqa: E402

from mshf.core.line_features import line_features, LINE_FEATURE_COLS  # noqa: E402
from mshf.core.geometric_features import geometric_features, GEOMETRIC_FEATURE_COLS  # noqa: E402

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

# category gốc trong hybrid_fusion_dataset.csv là chuỗi đầy đủ ("1.original"...).
_LABEL_FROM_HYBRID = config.CATEGORY_TO_LABEL


def _load_doc_level() -> pd.DataFrame:
    """Nạp nhánh A: 8 feature doc-level + key(doc_id) + label từ CSV gốc."""
    df = pd.read_csv(config.HYBRID_DATASET_CSV)
    missing = [c for c in config.DOC_FEATURE_COLS if c not in df.columns]
    if missing:
        raise SystemExit(f"CSV gốc thiếu cột feature doc-level: {missing}")
    df = df.rename(columns={"key": "source_document_id"})
    df["category"] = df["label"]  # label gốc là chuỗi category đầy đủ
    df["label"] = df["category"].map(_LABEL_FROM_HYBRID)
    df["is_tampered"] = (df["category"] != config.ORIGINAL_CAT).astype(int)
    return df


def build(max_per_cat: int = 0) -> pd.DataFrame:
    doc_df = _load_doc_level()
    rows: list[dict] = []
    for _, r in doc_df.iterrows():
        doc_id = str(r["source_document_id"])
        category = str(r["category"])
        if max_per_cat and sum(1 for x in rows if x["category"] == category) >= max_per_cat:
            continue
        row = {
            "sample_id": f"{doc_id}_{config.CATEGORY_TO_LABEL[category]}",
            "source_document_id": doc_id,
            "category": category,
            "label": r["label"],
            "is_tampered": int(r["is_tampered"]),
        }
        for c in config.DOC_FEATURE_COLS:
            row[c] = r[c]
        # Nhánh B + C: original tự so với chính nó -> feature 0 (không thay đổi).
        row.update(line_features(doc_id, category))
        row.update(geometric_features(doc_id, category))
        rows.append(row)
        if len(rows) % 100 == 0:
            print(f"  [{len(rows)} built] {category}/{doc_id}")
    df = pd.DataFrame(rows)
    return df


def main() -> None:
    ap = argparse.ArgumentParser(description="Build MSHF enhanced dataset (A+B+C)")
    ap.add_argument("--output", type=Path, default=config.OUTPUT_DIR / "enhanced_dataset.csv")
    ap.add_argument("--max-per-cat", type=int, default=0, help="Giới hạn số doc mỗi category (0=all)")
    args = ap.parse_args()

    print("MSHF build_dataset")
    print(f"  doc-level CSV : {config.HYBRID_DATASET_CSV}")
    print(f"  OCR text root : {config.OCR_TEXT_ROOT}")
    print(f"  layout root   : {config.LAYOUT_JSON_ROOT}\n")

    df = build(args.max_per_cat)

    feature_cols = config.DOC_FEATURE_COLS + LINE_FEATURE_COLS + GEOMETRIC_FEATURE_COLS
    meta_cols = ["sample_id", "source_document_id", "category", "label", "is_tampered"]
    df = df[meta_cols + feature_cols]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False, encoding="utf-8-sig")

    print(f"\nDONE: {len(df)} rows, {len(feature_cols)} features -> {args.output}")
    print("  label dist:", df["label"].value_counts().to_dict())
    print("  tampered  :", df["is_tampered"].value_counts().to_dict())
    print(f"  groups    : {df['source_document_id'].nunique()} source docs")


if __name__ == "__main__":
    main()
