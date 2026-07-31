# baseline_layoutlmv3.py
# =============================================
# Chạy: LOCAL (CPU) — chỉ đọc CSV đã tính sẵn
# Cần : pip install scikit-learn pandas
# Input: layout_embedding_similarity_report.csv (tuần 5)
# Output: result_layoutlmv3.csv
# =============================================
import argparse

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layout-csv", default="layout_embedding_similarity_report.csv")
    parser.add_argument("--output", default="result_layoutlmv3.csv")
    args = parser.parse_args()

    print("=== Baseline: LayoutLMv3 Embedding Similarity ===")
    df = pd.read_csv(args.layout_csv, encoding="utf-8-sig")

    # Chuẩn hóa tên cột
    if "attack_group" in df.columns:
        df = df.rename(columns={"attack_group": "category"})

    # Thêm hàng 1.original (so với chính nó → similarity = 1.0)
    n_orig = df[df["category"] == "2.insert"].shape[0]
    orig_rows = pd.DataFrame({
        "category": ["1.original"] * n_orig,
        "layoutlmv3_cosine_similarity": [1.0] * n_orig,
    })
    df_all = pd.concat([orig_rows, df[["category", "layoutlmv3_cosine_similarity"]]], ignore_index=True)
    print(f"Số mẫu sau khi ghép: {len(df_all)}")
    print(df_all["category"].value_counts().to_string())

    X = df_all[["layoutlmv3_cosine_similarity"]].values.astype(float)
    le = LabelEncoder()
    y = le.fit_transform(df_all["category"])

    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, C=1.0))
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(clf, X, y, cv=cv, scoring="accuracy")
    mean_acc, std_acc = float(scores.mean()), float(scores.std())

    print(f"Accuracy = {mean_acc:.4f} ± {std_acc:.4f} ({mean_acc*100:.2f}%)")

    pd.DataFrame([{
        "model": "LayoutLMv3 embedding similarity",
        "accuracy": round(mean_acc, 4),
        "std": round(std_acc, 4),
        "accuracy_pct": f"{mean_acc*100:.2f}%",
        "features": "LayoutLMv3 cosine embedding vs 1.original",
        "classifier": "Logistic Regression",
        "note": "Layout embedding kết hợp ảnh + text + bbox, dùng kết quả tuần 5"
    }]).to_csv(args.output, index=False)
    print(f"Đã ghi: {args.output}")


if __name__ == "__main__":
    main()