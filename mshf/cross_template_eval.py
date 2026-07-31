"""Cross-template evaluation — chia train/test theo template_id."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from mshf.models import SingleModel


def main():
    ap = argparse.ArgumentParser(description="Cross-template evaluation")
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
        default=config.OUTPUT_DIR / "cross_template",
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

    # Check template_id
    if "template_id" not in manifest.columns:
        print("Manifest khong co template_id - khong the chay cross-template")
        return

    # Filter valid template_id
    valid = manifest[
        manifest["template_id"].notna()
        & (manifest["template_id"] != "unknown")
        & (manifest["template_id"] != "")
    ]
    if len(valid) == 0:
        print("Khong co template_id that (chi co unknown/empty)")
        return

    unique_templates = valid["template_id"].unique()
    if len(unique_templates) < 2:
        print(f"Chi co {len(unique_templates)} template, can it nhat 2")
        return

    print(f"Templates: {len(unique_templates)}")

    # Join template_id vào dataset
    df["source_document_id"] = df["source_document_id"].astype(str)
    manifest["source_document_id"] = manifest["source_document_id"].astype(str)

    merged = df.merge(
        manifest[["source_document_id", "template_id"]].drop_duplicates(),
        on="source_document_id",
        how="left",
    )

    # Drop rows without template_id
    merged = merged[merged["template_id"].notna() & (merged["template_id"] != "unknown")]
    print(f"After join + filter: {len(merged)} rows")

    # Feature columns
    from mshf.line_features import LINE_FEATURE_COLS
    from mshf.geometric_features import GEOMETRIC_FEATURE_COLS

    feature_cols = [
        c
        for c in config.DOC_FEATURE_COLS + LINE_FEATURE_COLS + GEOMETRIC_FEATURE_COLS
        if c in merged.columns
    ]

    le = LabelEncoder()
    X = merged[feature_cols].to_numpy(float)
    y = le.fit_transform(merged[args.target].astype(str))
    groups = merged["template_id"].to_numpy()
    source_ids = merged["source_document_id"].to_numpy()

    # LeaveOneGroupOut theo template_id
    logo = LeaveOneGroupOut()
    fold_results = []

    args.out_dir.mkdir(parents=True, exist_ok=True)

    for fold_idx, (train_idx, test_idx) in enumerate(logo.split(X, y, groups), 1):
        template = groups[test_idx[0]]

        # Đảm bảo source-disjoint
        test_sources = set(source_ids[test_idx])
        clean_train = [i for i in train_idx if source_ids[i] not in test_sources]

        if len(clean_train) < 10 or len(test_idx) < 2:
            print(f"  SKIP fold {fold_idx} (template={template}): khong du du lieu")
            continue

        X_train, y_train = X[clean_train], y[clean_train]
        X_test, y_test = X[test_idx], y[test_idx]

        model = SingleModel(len(le.classes_))
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        f1 = float(f1_score(y_test, y_pred, average="macro"))
        ba = float(balanced_accuracy_score(y_test, y_pred))

        fold_results.append({
            "fold": fold_idx,
            "template_id": str(template),
            "train_count": len(clean_train),
            "test_count": len(test_idx),
            "macro_f1": f1,
            "balanced_accuracy": ba,
        })

        print(f"  fold {fold_idx} (template={template}): "
              f"train={len(clean_train)}, test={len(test_idx)}, "
              f"macro_f1={f1:.3f}")

        # Save fold predictions
        fold_pred = pd.DataFrame({
            "source_document_id": source_ids[test_idx],
            "template_id": groups[test_idx],
            "y_true": y_test,
            "y_pred": y_pred,
        })
        fold_pred.to_csv(
            args.out_dir / f"predictions_fold_{fold_idx}.csv",
            index=False,
            encoding="utf-8-sig",
        )

        # Save fold split info
        split_info = {
            "fold": fold_idx,
            "template_id": str(template),
            "train_sources": sorted(set(source_ids[clean_train].tolist())),
            "test_sources": sorted(set(source_ids[test_idx].tolist())),
        }
        (args.out_dir / f"splits_fold_{fold_idx}.json").write_text(
            json.dumps(split_info, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    # Save aggregate metrics
    if fold_results:
        results_df = pd.DataFrame(fold_results)
        results_df.to_csv(
            args.out_dir / "cross_template_metrics.csv",
            index=False,
            encoding="utf-8-sig",
        )

        print(f"\nAggregate:")
        print(f"  mean macro_f1: {results_df['macro_f1'].mean():.3f} "
              f"+/- {results_df['macro_f1'].std():.3f}")
        print(f"  mean balanced_acc: {results_df['balanced_accuracy'].mean():.3f}")
    else:
        print("\nKhong co ket qua nao.")

    # Save run config
    run_config = {
        "dataset": str(args.dataset),
        "manifest": str(args.manifest),
        "target": args.target,
        "feature_count": len(feature_cols),
        "templates": sorted([str(t) for t in unique_templates]),
        "total_folds": len(fold_results),
    }
    (args.out_dir / "run_config.json").write_text(
        json.dumps(run_config, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"\nDONE -> {args.out_dir}")


if __name__ == "__main__":
    main()