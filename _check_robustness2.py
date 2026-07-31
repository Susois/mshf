import pandas as pd
from pathlib import Path

base = Path("outputs/robustness")

# 1. Phân phối dự đoán
pred_file = base / "predictions_jpeg_1.csv"
if pred_file.exists():
    p = pd.read_csv(pred_file)
    print("=== predictions_jpeg_1 ===")
    print("y_pred counts:")
    print(p["y_pred"].value_counts())
    print("crosstab y_true vs y_pred:")
    print(pd.crosstab(p["y_true"], p["y_pred"]))
    print("proba_tampered describe:")
    print(p["proba_tampered"].describe())
else:
    print("(predictions_jpeg_1.csv chua ton tai)")

# 2. So sánh giá trị feature
clean = pd.read_csv("outputs/enhanced_dataset.csv")
j1 = pd.read_csv(base / "features_jpeg_1.csv")
j3 = pd.read_csv(base / "features_jpeg_3.csv")

feat_cols = [c for c in clean.select_dtypes("number").columns
             if c in j1.columns and c != "is_tampered"]

print("\n=== feature means (clean / jpeg1 / jpeg3) ===")
for c in feat_cols:
    print(f"  {c:35s} {clean[c].mean():12.6f} {j1[c].mean():12.6f} {j3[c].mean():12.6f}")

print("\n=== NaN / inf count ===")
for name, df in [("clean", clean), ("jpeg1", j1), ("jpeg3", j3)]:
    nan_c = int(df[feat_cols].isna().sum().sum())
    inf_c = int(((df[feat_cols] == float("inf")) | (df[feat_cols] == float("-inf"))).sum().sum())
    print(f"  {name}: NaN={nan_c}, inf={inf_c}")

# 3. Nhãn có giữ nguyên không
print("\n=== label check ===")
print("is_tampered clean counts:", clean["is_tampered"].astype(str).value_counts().to_dict())
print("is_tampered jpeg1 counts:", j1["is_tampered"].astype(str).value_counts().to_dict())
