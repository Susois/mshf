# train_hybrid_fusion.py
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from xgboost import XGBClassifier

df = pd.read_csv("hybrid_fusion_dataset.csv")

feature_cols = [
    "cer", "wer",
    "mean_similarity", "min_similarity", "std_similarity", "ref_to_hyp_mean", "hyp_to_ref_mean",
    "layoutlmv3_cosine_similarity",
]
X = df[feature_cols]

out_dir = Path("results")
out_dir.mkdir(exist_ok=True)
report_lines = []


def log(text=""):
    print(text)
    report_lines.append(text)


def run_classification(label_col: str, title: str, tag: str):
    y_raw = df[label_col]
    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    model = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.1, eval_metric="mlogloss")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    report_dict = classification_report(y_test, y_pred, target_names=le.classes_, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(cm, index=le.classes_, columns=le.classes_)
    importances = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)

    log(f"\n=== {title} ===")
    log(f"Accuracy = {acc:.4f}")
    log("\nClassification report:")
    log(classification_report(y_test, y_pred, target_names=le.classes_))
    log("Confusion matrix:")
    log(cm_df.to_string())
    log("\nFeature importance:")
    log(importances.to_string())

    # Lưu file riêng để dễ chèn vào Word/Excel
    cm_df.to_csv(out_dir / f"confusion_matrix_{tag}.csv")
    importances.to_csv(out_dir / f"feature_importance_{tag}.csv", header=["importance"])
    pd.DataFrame(report_dict).T.to_csv(out_dir / f"classification_report_{tag}.csv")

    return {"accuracy": acc, "tag": tag}


# --- Bản 5 lớp (loại tấn công) ---
result_multiclass = run_classification("label", "Accuracy (Proposed Method - 5 lớp)", "5class")

# --- Bản 2 lớp (authentic vs tampered) ---
df["binary_label"] = df["label"].apply(lambda x: "authentic" if x == "1.original" else "tampered")
result_binary = run_classification("binary_label", "Accuracy (Proposed Method - Binary)", "binary")

# --- File tổng hợp dạng text, dễ đọc ---
(out_dir / "report.txt").write_text("\n".join(report_lines), encoding="utf-8")

# --- File tổng hợp số liệu chính dạng JSON, dễ dùng lại cho báo cáo Word ---
summary = {
    "accuracy_5class": result_multiclass["accuracy"],
    "accuracy_binary": result_binary["accuracy"],
}
(out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

print(f"\nĐã ghi toàn bộ kết quả vào thư mục: {out_dir.resolve()}")