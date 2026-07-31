"""Phân tích kết quả theo severity và attack type."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import config


def main():
    ap = argparse.ArgumentParser(description="Severity analysis")
    ap.add_argument(
        "--predictions",
        type=Path,
        default=config.OUTPUT_DIR / "training" / "predictions_is_tampered_A_B1_C_single.csv",
    )
    ap.add_argument(
        "--manifest",
        type=Path,
        default=config.OUTPUT_DIR / "manifest" / "attack_manifest.csv",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=config.OUTPUT_DIR / "analysis",
    )
    args = ap.parse_args()

    # Load predictions
    pred = pd.read_csv(args.predictions)
    print(f"Loaded {len(pred)} predictions")

    # Load manifest
    if not args.manifest.exists():
        print(f"Manifest not found: {args.manifest}")
        return
    manifest = pd.read_csv(args.manifest)
    print(f"Loaded manifest: {len(manifest)} rows")

    # Check severity column
    if "severity" not in manifest.columns:
        print("Manifest khong co severity column - khong the phan tich")
        return

    # Check for unknown/empty severity
    valid_severities = manifest[
        manifest["severity"].notna() & (manifest["severity"] != "unknown") & (manifest["severity"] != "")
    ]
    if len(valid_severities) == 0:
        print("Khong co severity that (chi co unknown/empty) - khong the phan tich")
        return

    # Join predictions with manifest
    # Ensure source_document_id types match
    pred["source_document_id"] = pred["source_document_id"].astype(str)
    manifest["source_document_id"] = manifest["source_document_id"].astype(str)

    result = pred.merge(
        manifest[["source_document_id", "attack_type", "severity"]].drop_duplicates(),
        on="source_document_id",
        how="left",
    )

    print(f"After join: {len(result)} rows")
    missing = result["severity"].isna().sum()
    if missing > 0:
        print(f"  {missing} rows missing severity (original docs)")

    # Group analysis by attack_type + severity
    metrics_rows = []

    # Overall by severity
    for severity in ["low", "medium", "high"]:
        subset = result[result["severity"] == severity]
        if len(subset) == 0:
            continue

        metrics = {
            "group": "all",
            "severity": severity,
            "count": len(subset),
            "macro_f1": f1_score(subset.y_true, subset.y_pred, average="macro"),
            "balanced_accuracy": balanced_accuracy_score(subset.y_true, subset.y_pred),
            "precision": precision_score(
                subset.y_true, subset.y_pred, average="macro", zero_division=0
            ),
            "recall": recall_score(
                subset.y_true, subset.y_pred, average="macro", zero_division=0
            ),
        }
        metrics_rows.append(metrics)
        print(f"  {severity}: n={metrics['count']}, macro_f1={metrics['macro_f1']:.3f}")

    # By attack_type + severity
    for (att_type, sev), g in result.groupby(["attack_type", "severity"]):
        if pd.isna(sev) or sev == "unknown" or sev == "":
            continue
        if len(g) < 2:
            continue

        metrics = {
            "group": str(att_type),
            "severity": str(sev),
            "count": len(g),
            "macro_f1": f1_score(g.y_true, g.y_pred, average="macro"),
            "balanced_accuracy": balanced_accuracy_score(g.y_true, g.y_pred),
            "precision": precision_score(
                g.y_true, g.y_pred, average="macro", zero_division=0
            ),
            "recall": recall_score(
                g.y_true, g.y_pred, average="macro", zero_division=0
            ),
        }
        metrics_rows.append(metrics)
        print(f"  {att_type}/{sev}: n={metrics['count']}, macro_f1={metrics['macro_f1']:.3f}")

    # Save
    args.out_dir.mkdir(parents=True, exist_ok=True)

    df_metrics = pd.DataFrame(metrics_rows)
    df_metrics.to_csv(
        args.out_dir / "severity_metrics.csv", index=False, encoding="utf-8-sig"
    )

    result.to_csv(
        args.out_dir / "severity_predictions_joined.csv",
        index=False,
        encoding="utf-8-sig",
    )

    run_config = {
        "predictions": str(args.predictions),
        "manifest": str(args.manifest),
        "total_predictions": len(pred),
        "total_with_severity": int(result["severity"].notna().sum()),
    }
    (args.out_dir / "severity_run_config.json").write_text(
        json.dumps(run_config, indent=2), encoding="utf-8"
    )

    print(f"\nSaved to: {args.out_dir}")


if __name__ == "__main__":
    main()