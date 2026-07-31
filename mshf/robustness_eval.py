"""Đánh giá robustness trên corrupted features.

Logic: Dùng cross-validation (cùng splits như training gốc).
Mỗi fold: train trên clean features → predict trên perturbed features → tính metrics.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from mshf.models import SingleModel


def cross_validate_robustness(
    clean_df: pd.DataFrame,
    perturbed_df: pd.DataFrame,
    feature_cols: list[str],
    split_file: Path,
) -> dict:
    """
    Cross-validate: train trên clean features, predict trên perturbed features.

    Returns dict với aggregate metrics.
    """
    le = LabelEncoder()
    y_clean = le.fit_transform(clean_df["is_tampered"].astype(str))
    y_pert = le.transform(perturbed_df["is_tampered"].astype(str))

    X_clean = clean_df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0).to_numpy(float)
    X_pert = perturbed_df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0).to_numpy(float)

    # Load splits
    splits = pd.read_csv(split_file)
    mapping = splits.set_index("source_document_id")["fold"]

    clean_folds = clean_df["source_document_id"].astype(str).map(mapping)
    pert_folds = perturbed_df["source_document_id"].astype(str).map(mapping)

    all_pred = np.empty_like(y_pert)
    all_proba = np.zeros(len(y_pert))

    for fold in sorted(clean_folds.unique()):
        tr_idx = np.flatnonzero(clean_folds.to_numpy() != fold)
        te_idx = np.flatnonzero(pert_folds.to_numpy() == fold)

        if len(tr_idx) == 0 or len(te_idx) == 0:
            continue

        model = SingleModel(2)
        model.fit(X_clean[tr_idx], y_clean[tr_idx])

        all_pred[te_idx] = model.predict(X_pert[te_idx])
        proba = model.predict_proba(X_pert[te_idx])
        if proba.shape[1] == 2:
            all_proba[te_idx] = proba[:, 1]

    f1 = float(f1_score(y_pert, all_pred, average="macro"))
    ba = float(balanced_accuracy_score(y_pert, all_pred))

    result = {
        "macro_f1": round(f1, 4),
        "balanced_accuracy": round(ba, 4),
        "n_original": int((y_pert == 0).sum()),
        "n_tampered": int((y_pert == 1).sum()),
        "sample_count": len(y_pert),
    }

    try:
        result["auroc"] = round(float(roc_auc_score(y_pert, all_proba)), 4)
    except Exception:
        result["auroc"] = None

    # OOF predictions per sample để truy ngược
    pred_df = perturbed_df[["sample_id", "source_document_id", "category"]].copy()
    pred_df["y_true"] = y_pert
    pred_df["y_pred"] = all_pred
    pred_df["proba_tampered"] = all_proba

    return result, pred_df


def main():
    ap = argparse.ArgumentParser(description="Evaluate robustness (cross-validated)")
    ap.add_argument(
        "--dataset",
        type=Path,
        default=config.OUTPUT_DIR / "enhanced_dataset.csv",
    )
    ap.add_argument(
        "--splits",
        type=Path,
        default=config.OUTPUT_DIR / "audit" / "source_splits.csv",
    )
    ap.add_argument(
        "--features-dir",
        type=Path,
        default=config.OUTPUT_DIR / "robustness",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=config.OUTPUT_DIR / "robustness",
    )
    args = ap.parse_args()

    # Load clean dataset
    clean_df = pd.read_csv(args.dataset)
    print(f"Clean dataset: {clean_df.shape}")

    # Determine feature columns
    from mshf.line_features import LINE_FEATURE_COLS
    from mshf.geometric_features import GEOMETRIC_FEATURE_COLS
    feature_cols = [
        c for c in config.DOC_FEATURE_COLS + LINE_FEATURE_COLS + GEOMETRIC_FEATURE_COLS
        if c in clean_df.columns
    ]
    print(f"Features: {len(feature_cols)}")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Compute baseline (clean → clean, cross-validated)
    print("\nComputing baseline (clean features)...")
    baseline, baseline_pred = cross_validate_robustness(clean_df, clean_df, feature_cols, args.splits)
    baseline_f1 = baseline["macro_f1"]
    print(f"Baseline macro_f1: {baseline_f1:.4f}")
    baseline_pred.to_csv(args.out_dir / "predictions_clean.csv", index=False, encoding="utf-8-sig")

    corruptions = ["jpeg", "blur", "resize", "skew", "contrast", "noise", "perspective"]
    levels = [1, 2, 3, 4, 5]

    results = []

    for corruption in corruptions:
        for level in levels:
            features_file = args.features_dir / f"features_{corruption}_{level}.csv"
            if not features_file.exists():
                print(f"  SKIP {corruption}/{level}")
                continue

            try:
                perturbed_df = pd.read_csv(features_file)

                result, pred_df = cross_validate_robustness(
                    clean_df, perturbed_df, feature_cols, args.splits
                )

                drop = (baseline_f1 - result["macro_f1"]) / baseline_f1 * 100 if baseline_f1 > 0 else 0

                result.update({
                    "corruption": corruption,
                    "level": level,
                    "relative_drop_f1": round(drop, 2),
                })
                results.append(result)

                # Lưu OOF predictions cho corruption/level này
                pred_df.to_csv(
                    args.out_dir / f"predictions_{corruption}_{level}.csv",
                    index=False,
                    encoding="utf-8-sig",
                )

                print(
                    f"  {corruption}/{level}: "
                    f"macro_f1={result['macro_f1']:.4f}, "
                    f"drop={result['relative_drop_f1']:.1f}%"
                )
            except Exception as e:
                print(f"  FAIL {corruption}/{level}: {e}")

    # Save
    if results:
        df = pd.DataFrame(results)
        col_order = ["corruption", "level", "sample_count", "macro_f1",
                      "balanced_accuracy", "auroc", "relative_drop_f1"]
        df = df[[c for c in col_order if c in df.columns]]
        df.to_csv(args.out_dir / "robustness_matrix.csv", index=False, encoding="utf-8-sig")

        print(f"\nRobustness matrix saved")
        print(f"\nSummary:")
        print(df[["corruption", "level", "macro_f1", "relative_drop_f1"]].to_string(index=False))

    # Run config
    run_config = {
        "dataset": str(args.dataset),
        "splits": str(args.splits),
        "baseline_f1": baseline_f1,
        "total_results": len(results),
    }
    (args.out_dir / "run_config.json").write_text(
        json.dumps(run_config, indent=2), encoding="utf-8"
    )
    print(f"\nDONE -> {args.out_dir}")


if __name__ == "__main__":
    main()