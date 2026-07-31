import pandas as pd

p = pd.read_csv("outputs/robustness/predictions_jpeg_with_level" if False else "outputs/robustness/predictions_jpeg_1.csv")
print("y_pred distribution:")
print(p["y_pred"].value_counts())
print("\ny_true distribution:")
print(p["y_true"].value_counts())
print("\nproba_tampered stats:")
print(p["proba_tampered"].describe())
