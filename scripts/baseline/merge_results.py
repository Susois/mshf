# merge_results.py
# =============================================
# Chạy: LOCAL (CPU) — chạy SAU KHI đã có đủ các file result_*.csv
# Cần : pip install pandas
# Input: result_levenshtein.csv, result_tfidf.csv, result_phobert.csv,
#        result_sbert.csv (nếu có), result_layoutlmv3.csv, result_proposed.csv
# Output: baseline_comparison.csv  (bảng so sánh cuối cùng cho báo cáo)
# =============================================
import argparse
from pathlib import Path
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", default=".",
                        help="Thư mục chứa các file result_*.csv")
    parser.add_argument("--output", default="baseline_comparison.csv")
    args = parser.parse_args()

    result_dir = Path(args.result_dir)

    # Thứ tự hiển thị trong bảng
    FILE_ORDER = [
        ("result_levenshtein.csv", "Levenshtein"),
        ("result_tfidf.csv", "TF-IDF"),
        ("result_phobert.csv", "PhoBERT"),
        ("result_sbert.csv", "SBERT"),          # có thể chưa có nếu chưa chạy Colab
        ("result_layoutlmv3.csv", "LayoutLMv3"),
        ("result_proposed.csv", "Proposed"),
    ]

    rows = []
    for filename, label in FILE_ORDER:
        path = result_dir / filename
        if not path.exists():
            print(f"[WARN] Chưa có: {filename} — bỏ qua (có thể chưa chạy SBERT trên Colab)")
            continue
        df = pd.read_csv(path)
        rows.append(df.iloc[0].to_dict())

    if not rows:
        print("[ERROR] Không tìm thấy file kết quả nào!")
        return

    df_final = pd.DataFrame(rows).sort_values("accuracy", ascending=False).reset_index(drop=True)
    df_final.index += 1  # Bắt đầu từ 1

    # In bảng đẹp ra terminal
    print("\n" + "="*80)
    print("BẢNG SO SÁNH ACCURACY — TUẦN 6")
    print("="*80)
    print(f"{'#':<4} {'Model':<40} {'Accuracy':<12} {'±Std':<10} {'Classifier'}")
    print("-"*80)
    for i, row in df_final.iterrows():
        print(f"{i:<4} {row['model']:<40} {row['accuracy_pct']:<12} ±{row['std']*100:.2f}%   {row['classifier']}")
    print("="*80)

    df_final.to_csv(args.output, index=False, encoding="utf-8-sig")
    print(f"\nĐã ghi bảng so sánh đầy đủ: {args.output}")

    # Gợi ý phân tích nhanh
    best = df_final.iloc[0]
    proposed = df_final[df_final["model"].str.contains("Proposed")].iloc[0] if any(df_final["model"].str.contains("Proposed")) else None
    print(f"\nModel tốt nhất: {best['model']} ({best['accuracy_pct']})")
    if proposed is not None:
        print(f"Proposed Method: {proposed['accuracy_pct']} (xếp hạng #{df_final[df_final['model'].str.contains('Proposed')].index[0]})")


if __name__ == "__main__":
    main()