"""Extract features cho corrupted controls bằng feature perturbation.

Vì corrupted PDFs là image-only (không có text layer), không thể dùng
PyMuPDF text extraction. Thay vào đó, mô phỏng ảnh hưởng của corruption
lên features bằng cách thêm nhiễu có kiểm soát vào features gốc.

Mỗi loại corruption ảnh hưởng khác nhau:
- JPEG/blur/noise → tăng CER/WER, giảm similarity (OCR kém hơn)
- Resize → ảnh hưởng vừa phải
- Skew/perspective → ảnh hưởng layout features nhiều hơn
- Contrast → ảnh hưởng nhẹ (OCR thường robust với contrast)

Level 1-5 tương ứng mức nhiễu tăng dần.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import config
from mshf.core.line_features import LINE_FEATURE_COLS
from mshf.core.geometric_features import GEOMETRIC_FEATURE_COLS


# Hệ số ảnh hưởng của từng corruption lên từng nhóm feature
# (ocr_impact, line_impact, geo_impact) ở level=1, scale tuyến tính theo level
CORRUPTION_IMPACT = {
    "jpeg":        {"ocr": 0.02, "line": 0.01, "geo": 0.005},
    "blur":        {"ocr": 0.05, "line": 0.03, "geo": 0.005},
    "resize":      {"ocr": 0.03, "line": 0.02, "geo": 0.01},
    "skew":        {"ocr": 0.04, "line": 0.02, "geo": 0.05},
    "contrast":    {"ocr": 0.01, "line": 0.005, "geo": 0.002},
    "noise":       {"ocr": 0.06, "line": 0.04, "geo": 0.005},
    "perspective": {"ocr": 0.05, "line": 0.03, "geo": 0.06},
}


def perturb_features(
    original_df: pd.DataFrame,
    corruption: str,
    level: int,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Tạo corrupted features bằng cách perturb features gốc.

    Perturb TẤT CẢ documents (cả original và tampered) vì corruption
    ảnh hưởng toàn bộ quá trình scan/copy, không chỉ original.
    """
    rng = np.random.default_rng(seed + level + hash(corruption) % 10000)
    impact = CORRUPTION_IMPACT.get(corruption, {"ocr": 0.03, "line": 0.02, "geo": 0.01})

    # Perturb TẤT CẢ rows (cả original và tampered)
    df = original_df.copy()

    # Scale theo level (level 1 = nhẹ, level 5 = nặng)
    scale = level

    # Perturb Branch A (OCR features)
    ocr_cols = [c for c in config.DOC_FEATURE_COLS if c in df.columns]
    for col in ocr_cols:
        values = df[col].to_numpy(float)
        noise_std = impact["ocr"] * scale
        if col in ("cer", "wer"):
            # CER/WER tăng khi corruption tăng
            noise = rng.uniform(0, noise_std * 2, len(values))
            df[col] = np.clip(values + noise, 0, 1)
        elif "similarity" in col or "mean" in col.lower():
            # Similarity giảm khi corruption tăng
            noise = rng.uniform(0, noise_std * 2, len(values))
            df[col] = np.clip(values - noise, 0, 1)
        else:
            # Thêm Gaussian noise
            noise = rng.normal(0, noise_std, len(values))
            df[col] = values + noise

    # Perturb Branch B1 (line features)
    line_cols = [c for c in LINE_FEATURE_COLS if c in df.columns]
    for col in line_cols:
        values = df[col].to_numpy(float)
        noise_std = impact["line"] * scale
        if "ratio" in col:
            noise = rng.normal(0, noise_std * 0.5, len(values))
            df[col] = np.clip(values + noise, 0, 1)
        elif "count" in col:
            # Count features: thêm một ít
            noise = rng.poisson(noise_std * 2, len(values))
            df[col] = np.clip(values + noise, 0, None)
        else:
            noise = rng.normal(0, noise_std, len(values))
            df[col] = values + noise

    # Perturb Branch C (geometric features)
    geo_cols = [c for c in GEOMETRIC_FEATURE_COLS if c in df.columns]
    for col in geo_cols:
        values = df[col].to_numpy(float)
        noise_std = impact["geo"] * scale
        noise = rng.normal(0, noise_std, len(values))
        df[col] = values + noise

    # Update sample_id
    df["sample_id"] = df["sample_id"].astype(str) + f"_{corruption}_{level}"

    return df


def main():
    ap = argparse.ArgumentParser(
        description="Extract features cho corrupted controls (feature perturbation)"
    )
    ap.add_argument(
        "--dataset",
        type=Path,
        default=config.OUTPUT_DIR / "enhanced_dataset.csv",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=config.OUTPUT_DIR / "robustness",
    )
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    # Load original features
    original_features = pd.read_csv(args.dataset)
    print(f"Loaded original features: {original_features.shape}")
    print(f"  Original docs: {(original_features['is_tampered'] == 0).sum()}")
    print(f"  Tampered docs: {(original_features['is_tampered'] == 1).sum()}")

    corruptions = list(CORRUPTION_IMPACT.keys())
    levels = [1, 2, 3, 4, 5]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    total = 0

    for corruption in corruptions:
        for level in levels:
            df = perturb_features(
                original_features, corruption, level, seed=args.seed
            )

            output_path = args.out_dir / f"features_{corruption}_{level}.csv"
            df.to_csv(output_path, index=False, encoding="utf-8-sig")
            total += 1
            print(f"  {corruption}/{level}: {len(df)} rows -> {output_path.name}")

    print(f"\nDONE: {total} feature files -> {args.out_dir}")
    print("Tiep theo chay: python -m mshf.robustness_eval")


if __name__ == "__main__":
    main()