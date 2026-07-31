"""Tạo calibration plot, case studies và failure taxonomy."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import config
from mshf.core.evaluate import expected_calibration_error


def find_proba_column(df: pd.DataFrame) -> str:
    """Tìm cột probability cho class positive (tampered/1)."""
    for col in ["proba_1", "proba_tampered", "proba_True"]:
        if col in df.columns:
            return col
    # Fallback: tìm cột bắt đầu bằng proba_ mà không phải class 0
    proba_cols = [c for c in df.columns if c.startswith("proba_")]
    if len(proba_cols) == 2:
        # Binary: trả về cột thứ 2 (class 1)
        return proba_cols[1]
    if len(proba_cols) >= 1:
        return proba_cols[-1]
    raise ValueError(f"Khong tim thay cot probability. Columns: {df.columns.tolist()}")


def create_calibration_plot(pred_df: pd.DataFrame, proba_col: str, output_path: Path):
    """Tạo calibration plot từ OOF predictions."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib chua cai dat, bo qua calibration plot")
        return

    y_true = pred_df["y_true"].to_numpy()
    y_prob = pred_df[proba_col].to_numpy()

    prob_true, prob_pred = calibration_curve(
        y_true, y_prob, n_bins=10, strategy="uniform"
    )

    plt.figure(figsize=(8, 6))
    plt.plot(prob_pred, prob_true, "o-", linewidth=2, label="Mo hinh")
    plt.plot([0, 1], [0, 1], "k--", linewidth=1, label="Hoan hao")
    plt.xlabel("Xac suat du doan trung binh")
    plt.ylabel("Ty le positives thuc")
    plt.title("Bieu do Calibration - Phat hien gia mao")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  Calibration plot saved: {output_path}")

    # Save calibration bins
    bins_df = pd.DataFrame({
        "mean_predicted_prob": prob_pred,
        "fraction_of_positives": prob_true,
    })
    bins_path = output_path.parent / "calibration_bins.csv"
    bins_df.to_csv(bins_path, index=False, encoding="utf-8-sig")

    # Calibration metrics
    brier = float(brier_score_loss(y_true, y_prob))
    ece = expected_calibration_error(y_true, y_prob)
    print(f"  Brier score: {brier:.4f}")
    print(f"  ECE: {ece:.4f}")

    return {"brier": brier, "ece": ece}


def create_case_studies(
    pred_df: pd.DataFrame, proba_col: str, output_path: Path, n_per_group: int = 5
):
    """Chọn TP, FP, FN đại diện."""
    cases = []

    y_prob = pred_df[proba_col]

    # True Positives (đúng tấn công)
    tp = pred_df[(pred_df["y_true"] == 1) & (pred_df["y_pred"] == 1)]
    tp_sorted = tp.sort_values(proba_col, ascending=False)
    for _, row in tp_sorted.head(n_per_group).iterrows():
        cases.append({
            "case_type": "TP",
            "sample_id": row.get("sample_id", ""),
            "source_document_id": row["source_document_id"],
            "y_true": int(row["y_true"]),
            "y_pred": int(row["y_pred"]),
            "confidence": float(row[proba_col]),
            "reason": "Phat hien dung tan cong",
        })

    # False Positives (cảnh báo sai)
    fp = pred_df[(pred_df["y_true"] == 0) & (pred_df["y_pred"] == 1)]
    fp_sorted = fp.sort_values(proba_col, ascending=False)
    for _, row in fp_sorted.head(n_per_group).iterrows():
        cases.append({
            "case_type": "FP",
            "sample_id": row.get("sample_id", ""),
            "source_document_id": row["source_document_id"],
            "y_true": int(row["y_true"]),
            "y_pred": int(row["y_pred"]),
            "confidence": float(row[proba_col]),
            "reason": "Canh bao sai - tai lieu sach",
        })

    # False Negatives (bỏ sót)
    fn = pred_df[(pred_df["y_true"] == 1) & (pred_df["y_pred"] == 0)]
    fn_sorted = fn.sort_values(proba_col, ascending=True)
    for _, row in fn_sorted.head(n_per_group).iterrows():
        cases.append({
            "case_type": "FN",
            "sample_id": row.get("sample_id", ""),
            "source_document_id": row["source_document_id"],
            "y_true": int(row["y_true"]),
            "y_pred": int(row["y_pred"]),
            "confidence": float(row[proba_col]),
            "reason": "Bo sot tan cong",
        })

    # True Negatives (đúng sạch) — chọn vài ví dụ
    tn = pred_df[(pred_df["y_true"] == 0) & (pred_df["y_pred"] == 0)]
    tn_sorted = tn.sort_values(proba_col, ascending=True)
    for _, row in tn_sorted.head(min(3, n_per_group)).iterrows():
        cases.append({
            "case_type": "TN",
            "sample_id": row.get("sample_id", ""),
            "source_document_id": row["source_document_id"],
            "y_true": int(row["y_true"]),
            "y_pred": int(row["y_pred"]),
            "confidence": float(row[proba_col]),
            "reason": "Dung xac nhan tai lieu sach",
        })

    df = pd.DataFrame(cases)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"  Case studies saved: {output_path} ({len(cases)} cases)")


def classify_failure_cause(row, proba_col: str) -> str:
    """Phân loại nguyên nhân lỗi dựa trên heuristic."""
    if row["y_true"] == 0 and row["y_pred"] == 1:
        return "OCR degradation tren authentic"
    elif row["y_true"] == 1 and row["y_pred"] == 0:
        conf = row[proba_col]
        if conf < 0.3:
            return "Low semantic difference"
        elif conf < 0.5:
            return "Borderline confidence"
        else:
            return "Feature extraction failure"
    return "Unknown"


def create_failure_taxonomy(
    pred_df: pd.DataFrame,
    proba_col: str,
    localization_path: Path,
    output_path: Path,
):
    """Tạo taxonomy của các lỗi dự đoán."""
    # Load localization details nếu có
    loc_details = {}
    if localization_path.exists():
        try:
            with open(localization_path, encoding="utf-8") as f:
                raw = json.load(f)
            # Hỗ trợ cả dict và list format
            if isinstance(raw, dict):
                loc_details = raw
            elif isinstance(raw, list):
                # Convert list to dict keyed by source_document_id
                for item in raw:
                    if isinstance(item, dict):
                        key = str(item.get("source_document_id", item.get("sample_id", "")))
                        if key:
                            loc_details[key] = item
        except (json.JSONDecodeError, OSError):
            pass

    # Tìm tất cả errors
    errors = pred_df[pred_df["y_true"] != pred_df["y_pred"]]

    failures = []
    for _, row in errors.iterrows():
        error_type = "False Positive" if row["y_true"] == 0 else "False Negative"
        doc_id = str(row["source_document_id"])
        loc_info = loc_details.get(doc_id, {})

        failures.append({
            "sample_id": row.get("sample_id", ""),
            "source_document_id": doc_id,
            "error_type": error_type,
            "y_true": int(row["y_true"]),
            "y_pred": int(row["y_pred"]),
            "confidence": float(row[proba_col]),
            "possible_cause": classify_failure_cause(row, proba_col),
            "localization_available": bool(loc_info),
        })

    df = pd.DataFrame(failures)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"  Failure taxonomy saved: {output_path} ({len(failures)} errors)")


def main():
    ap = argparse.ArgumentParser(description="Tao bao cao va visualizations")
    ap.add_argument(
        "--predictions",
        type=Path,
        default=config.OUTPUT_DIR / "training" / "predictions_is_tampered_A_B1_C_stacking.csv",
    )
    ap.add_argument(
        "--localization",
        type=Path,
        default=config.OUTPUT_DIR / "localization" / "localization_details.json",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=config.OUTPUT_DIR / "report",
    )
    ap.add_argument(
        "--n-cases",
        type=int,
        default=5,
        help="So case study moi nhom (TP, FP, FN)",
    )
    args = ap.parse_args()

    # Load predictions
    if not args.predictions.exists():
        # Tìm file predictions khác
        training_dir = config.OUTPUT_DIR / "training"
        alternatives = sorted(training_dir.glob("predictions_is_tampered_*.csv"))
        if alternatives:
            args.predictions = alternatives[0]
            print(f"Dung file thay the: {args.predictions}")
        else:
            print(f"Khong tim thay file predictions")
            return

    pred_df = pd.read_csv(args.predictions)
    print(f"Loaded {len(pred_df)} predictions from {args.predictions.name}")

    # Tìm probability column
    try:
        proba_col = find_proba_column(pred_df)
        print(f"Probability column: {proba_col}")
    except ValueError as e:
        print(str(e))
        return

    # Create output directory
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Calibration plot
    print("\n--- Calibration Plot ---")
    cal_metrics = create_calibration_plot(
        pred_df, proba_col, args.out_dir / "calibration_plot.png"
    )

    # 2. Case studies
    print("\n--- Case Studies ---")
    create_case_studies(
        pred_df, proba_col, args.out_dir / "case_studies.csv", n_per_group=args.n_cases
    )

    # 3. Failure taxonomy
    print("\n--- Failure Taxonomy ---")
    create_failure_taxonomy(
        pred_df, proba_col, args.localization, args.out_dir / "failure_taxonomy.csv"
    )

    # 4. Summary statistics
    tp = len(pred_df[(pred_df["y_true"] == 1) & (pred_df["y_pred"] == 1)])
    tn = len(pred_df[(pred_df["y_true"] == 0) & (pred_df["y_pred"] == 0)])
    fp = len(pred_df[(pred_df["y_true"] == 0) & (pred_df["y_pred"] == 1)])
    fn = len(pred_df[(pred_df["y_true"] == 1) & (pred_df["y_pred"] == 0)])

    summary = {
        "predictions_file": str(args.predictions),
        "total_samples": len(pred_df),
        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "accuracy": (tp + tn) / len(pred_df) if len(pred_df) > 0 else 0,
        "proba_column": proba_col,
    }
    if cal_metrics:
        summary.update(cal_metrics)

    (args.out_dir / "run_config.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # report_summary.md — số liệu truy ngược được tới predictions gốc
    lines = [
        "# Bao cao tong hop (Yeu cau 26)",
        "",
        f"- Predictions: `{args.predictions}`",
        f"- Tong so mau: {len(pred_df)}",
        f"- TP={tp} | TN={tn} | FP={fp} | FN={fn}",
        f"- Accuracy: {summary['accuracy']:.4f}",
    ]
    if cal_metrics:
        lines += [
            f"- Brier score: {cal_metrics['brier']:.4f}",
            f"- ECE: {cal_metrics['ece']:.4f}",
        ]
    lines += [
        "",
        "## Artifacts",
        "",
        "- `calibration_plot.png` / `calibration_bins.csv` — reliability diagram tu OOF probability",
        "- `case_studies.csv` — TP/FP/FN/TN dai dien kem confidence",
        "- `failure_taxonomy.csv` — phan loai loi kem possible_cause",
        "- `run_config.json` — cau hinh va thong ke run",
    ]
    (args.out_dir / "report_summary.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )

    print(f"\nSummary: TP={tp}, TN={tn}, FP={fp}, FN={fn}")
    print(f"DONE -> {args.out_dir}")


if __name__ == "__main__":
    main()