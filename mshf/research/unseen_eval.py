"""Unseen-subtype/generator evaluation — leave-one-out theo subtype hoặc generator."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import config
from mshf.core.models import SingleModel


def run_holdout(
    dataset: pd.DataFrame,
    feature_cols: list[str],
    target: str,
    group_col: str,
    holdout_value: str,
) -> dict:
    """
    Train trên tất cả trừ holdout_value, test trên holdout_value.

    Args:
        dataset: DataFrame đầy đủ features + labels + metadata
        feature_cols: danh sách cột feature
        target: 'is_tampered' hoặc 'label'
        group_col: 'attack_subtype' hoặc 'generator_id'
        holdout_value: giá trị giữ lại cho test

    Returns:
        dict metrics
    """
    train_mask = dataset[group_col] != holdout_value
    test_mask = dataset[group_col] == holdout_value

    # Đảm bảo không có source document rò rỉ
    test_sources = set(dataset.loc[test_mask, "source_document_id"].unique())
    train_mask = train_mask & ~dataset["source_document_id"].isin(test_sources)

    train_df = dataset[train_mask]
    test_df = dataset[test_mask]

    if len(train_df) < 10 or len(test_df) < 2:
        return None

    le = LabelEncoder()
    X_train = train_df[feature_cols].to_numpy(float)
    y_train = le.fit_transform(train_df[target].astype(str))
    X_test = test_df[feature_cols].to_numpy(float)
    y_test = le.transform(test_df[target].astype(str))

    model = SingleModel(len(le.classes_))
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    metrics = {
        "holdout_group": group_col,
        "holdout_value": holdout_value,
        "target": target,
        "train_count": len(train_df),
        "test_count": len(test_df),
        "train_sources": int(train_df["source_document_id"].nunique()),
        "test_sources": int(test_df["source_document_id"].nunique()),
        "macro_f1": float(f1_score(y_test, y_pred, average="macro")),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred)),
    }

    # Predictions per sample của tập holdout
    pred_df = test_df[["sample_id", "source_document_id", "category"]].copy()
    pred_df["y_true"] = y_test
    pred_df["y_pred"] = y_pred
    for i, cls in enumerate(le.classes_):
        pred_df[f"proba_{cls}"] = y_proba[:, i]

    # Split info để kiểm tra không rò rỉ
    split_info = {
        "holdout_group": group_col,
        "holdout_value": str(holdout_value),
        "train_sources": sorted(train_df["source_document_id"].astype(str).unique().tolist()),
        "test_sources": sorted(test_df["source_document_id"].astype(str).unique().tolist()),
    }

    return metrics, pred_df, split_info


def main():
    ap = argparse.ArgumentParser(description="Unseen-subtype/generator evaluation")
    ap.add_argument(
        "--dataset",
        type=Path,
        default=config.OUTPUT_DIR / "enhanced_dataset.csv",
    )
    ap.add_argument(
        "--manifest",
        type=Path,
        default=config.OUTPUT_DIR / "manifest" / "attack_manifest.csv",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=config.OUTPUT_DIR / "unseen",
    )
    ap.add_argument(
        "--target",
        choices=["is_tampered", "label"],
        default="is_tampered",
    )
    args = ap.parse_args()

    # Load dataset
    df = pd.read_csv(args.dataset)
    print(f"Dataset: {len(df)} rows")

    # Load manifest
    if not args.manifest.exists():
        print(f"Manifest not found: {args.manifest}")
        return
    manifest = pd.read_csv(args.manifest)
    print(f"Manifest: {len(manifest)} rows")

    # Join manifest metadata
    df["source_document_id"] = df["source_document_id"].astype(str)
    manifest["source_document_id"] = manifest["source_document_id"].astype(str)

    merge_cols = ["source_document_id"]
    extra_cols = []
    for col in ["attack_subtype", "generator_id"]:
        if col in manifest.columns:
            extra_cols.append(col)
    if not extra_cols:
        print("Manifest khong co attack_subtype hoac generator_id")
        return

    merged = df.merge(
        manifest[merge_cols + extra_cols].drop_duplicates(),
        on="source_document_id",
        how="left",
    )

    # Determine feature columns
    from mshf.core.line_features import LINE_FEATURE_COLS
    from mshf.core.geometric_features import GEOMETRIC_FEATURE_COLS

    feature_cols = [
        c
        for c in config.DOC_FEATURE_COLS + LINE_FEATURE_COLS + GEOMETRIC_FEATURE_COLS
        if c in merged.columns
    ]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    all_results = []

    # Run for each group column
    for group_col in extra_cols:
        valid = merged[
            merged[group_col].notna()
            & (merged[group_col] != "unknown")
            & (merged[group_col] != "")
        ]
        unique_values = valid[group_col].unique()

        if len(unique_values) < 2:
            print(f"  {group_col}: chi co {len(unique_values)} gia tri, can it nhat 2")
            continue

        print(f"\n--- Leave-one-out theo {group_col} ({len(unique_values)} groups) ---")

        for holdout in unique_values:
            result = run_holdout(
                merged, feature_cols, args.target, group_col, holdout
            )
            if result is None:
                print(f"  SKIP {holdout}: khong du du lieu")
                continue

            metrics, pred_df, split_info = result
            all_results.append(metrics)
            print(
                f"  {holdout}: train={metrics['train_count']}, "
                f"test={metrics['test_count']}, "
                f"macro_f1={metrics['macro_f1']:.3f}"
            )

            # Save predictions và split của holdout này
            safe_value = str(holdout).replace("/", "_").replace(" ", "_")
            pred_df.to_csv(
                args.out_dir / f"predictions_holdout_{group_col}_{safe_value}.csv",
                index=False,
                encoding="utf-8-sig",
            )
            (args.out_dir / f"splits_holdout_{group_col}_{safe_value}.json").write_text(
                json.dumps(split_info, indent=2, ensure_ascii=False), encoding="utf-8"
            )

    # Save aggregate results
    if all_results:
        results_df = pd.DataFrame(all_results)
        results_df.to_csv(
            args.out_dir / "unseen_metrics.csv", index=False, encoding="utf-8-sig"
        )

        # Tách riêng metrics theo subtype / generator
        for group_col in results_df["holdout_group"].unique():
            subset = results_df[results_df["holdout_group"] == group_col]
            subset.to_csv(
                args.out_dir / f"unseen_{group_col.replace('attack_', '').replace('_id', '')}_metrics.csv",
                index=False,
                encoding="utf-8-sig",
            )
            print(f"\n{group_col} summary:")
            print(f"  mean macro_f1: {subset['macro_f1'].mean():.3f} +/- {subset['macro_f1'].std():.3f}")
            print(f"  mean balanced_acc: {subset['balanced_accuracy'].mean():.3f}")
    else:
        print("\nKhong co ket qua nao.")

    # Save run config
    run_config = {
        "dataset": str(args.dataset),
        "manifest": str(args.manifest),
        "target": args.target,
        "feature_count": len(feature_cols),
        "total_holdouts": len(all_results),
    }
    (args.out_dir / "run_config.json").write_text(
        json.dumps(run_config, indent=2), encoding="utf-8"
    )

    print(f"\nDONE -> {args.out_dir}")


if __name__ == "__main__":
    main()