# baseline_phobert.py
# =============================================
# Chạy: LOCAL (CPU) — chỉ đọc CSV đã tính sẵn
# Cần : pip install scikit-learn pandas
# Input: semantic_similarity_vs_original.csv (tuần 4 đã hiệu chỉnh)
# Output: result_phobert.csv
# =============================================
import argparse

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

FEATURE_COLS = [
    "mean_similarity", "min_similarity", "std_similarity",
    "ref_to_hyp_mean", "hyp_to_ref_mean",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--semantic-csv", default="semantic_similarity_vs_original.csv")
    parser.add_argument("--output", default="result_phobert.csv")
    args = parser.parse_args()

    print("=== Baseline: PhoBERT Semantic Similarity ===")
    df = pd.read_csv(args.semantic_csv, encoding="utf-8-sig")
    df = df[df["status"] == "ok"].reset_index(drop=True)
    print(f"Số mẫu: {len(df)}")
    print(df["category"].value_counts().to_string())

    X = df[FEATURE_COLS].values.astype(float)
    le = LabelEncoder()
    y = le.fit_transform(df["category"])

    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, C=1.0))
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(clf, X, y, cv=cv, scoring="accuracy")
    mean_acc, std_acc = float(scores.mean()), float(scores.std())

    print(f"Accuracy = {mean_acc:.4f} ± {std_acc:.4f} ({mean_acc*100:.2f}%)")

    pd.DataFrame([{
        "model": "PhoBERT semantic similarity",
        "accuracy": round(mean_acc, 4),
        "std": round(std_acc, 4),
        "accuracy_pct": f"{mean_acc*100:.2f}%",
        "features": "mean/min/std similarity + ref↔hyp mean (PhoBERT cosine)",
        "classifier": "Logistic Regression",
        "note": "Embedding ngữ nghĩa tiếng Việt, dùng kết quả đã tính tuần 4"
    }]).to_csv(args.output, index=False)
    print(f"Đã ghi: {args.output}")


if __name__ == "__main__":
    main()