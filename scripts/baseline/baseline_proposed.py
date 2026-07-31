# baseline_proposed.py
# =============================================
# Chạy: LOCAL (CPU)
# Cần : pip install xgboost scikit-learn pandas
# Input: hybrid_fusion_dataset.csv (tuần 5)
# Output: result_proposed.csv
# =============================================
import argparse
import pickle

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from xgboost import XGBClassifier

FEATURE_COLS = [
    "cer", "wer",
    "mean_similarity", "min_similarity", "std_similarity",
    "ref_to_hyp_mean", "hyp_to_ref_mean",
    "layoutlmv3_cosine_similarity",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hybrid-csv", default="hybrid_fusion_dataset.csv")
    parser.add_argument("--output", default="result_proposed.csv")
    parser.add_argument("--save-model", default="hybrid_model.pkl",
                        help="Lưu model đã train để dùng trong detector.py / explainer.py")
    args = parser.parse_args()

    print("=== Proposed Method: Hybrid Fusion (XGBoost) ===")
    df = pd.read_csv(args.hybrid_csv)
    print(f"Số mẫu: {len(df)}")
    print(df["label"].value_counts().to_string())

    X = df[FEATURE_COLS].values.astype(float)
    le = LabelEncoder()
    y = le.fit_transform(df["label"])

    clf = XGBClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.1,
        eval_metric="mlogloss", random_state=42,
    )

    # 5-fold cross-validation (để so sánh công bằng với các baseline)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(clf, X, y, cv=cv, scoring="accuracy")
    mean_acc, std_acc = float(scores.mean()), float(scores.std())
    print(f"CV Accuracy = {mean_acc:.4f} ± {std_acc:.4f} ({mean_acc*100:.2f}%)")

    # Train lại trên toàn bộ dữ liệu để lưu model phục vụ detector/explainer
    clf.fit(X, y)
    with open(args.save_model, "wb") as f:
        pickle.dump(clf, f)
    with open("label_encoder.pkl", "wb") as f:
        pickle.dump(le, f)
    print(f"Model đã lưu: {args.save_model}")
    print(f"Label encoder đã lưu: label_encoder.pkl")

    pd.DataFrame([{
        "model": "Proposed (Hybrid Fusion)",
        "accuracy": round(mean_acc, 4),
        "std": round(std_acc, 4),
        "accuracy_pct": f"{mean_acc*100:.2f}%",
        "features": "CER+WER + PhoBERT (5 feature) + LayoutLMv3 cosine",
        "classifier": "XGBoost",
        "note": "Kết hợp 3 nhánh: OCR + Semantic + Layout"
    }]).to_csv(args.output, index=False)
    print(f"Đã ghi: {args.output}")

    
if __name__ == "__main__":
    main()