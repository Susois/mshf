# baseline_levenshtein.py
# =============================================
# Chạy: LOCAL (CPU)
# Cần : pip install rapidfuzz scikit-learn
# Input: thư mục 3.ocr_output/
# Output: result_levenshtein.csv
# =============================================
import argparse
import re
import unicodedata
from pathlib import Path

import pandas as pd
from rapidfuzz.distance import Levenshtein
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

CATEGORIES = ["1.original", "2.insert", "3.delete", "4.modify", "5.layout"]


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    return re.sub(r"\s+", " ", text).strip().lower()


def compute_features(ocr_dir: Path) -> tuple:
    original_dir = ocr_dir / "1.original"
    X_rows, labels = [], []

    for cat in CATEGORIES:
        cat_dir = ocr_dir / cat
        if not cat_dir.exists():
            print(f"[WARN] Không thấy: {cat_dir}")
            continue
        txt_files = sorted(cat_dir.rglob("*.txt"))
        print(f"  [{cat}] {len(txt_files)} file")
        for txt_path in txt_files:
            rel = txt_path.relative_to(cat_dir)
            orig_path = original_dir / rel
            if not orig_path.exists():
                continue
            ref = normalize(orig_path.read_text(encoding="utf-8-sig", errors="ignore"))
            hyp = normalize(txt_path.read_text(encoding="utf-8-sig", errors="ignore"))
            ref_words = ref.split()
            hyp_words = hyp.split()
            cer = Levenshtein.distance(ref, hyp) / max(len(ref), 1)
            wer = Levenshtein.distance(ref_words, hyp_words) / max(len(ref_words), 1)
            X_rows.append([cer, wer])
            labels.append(cat)

    return X_rows, labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ocr-dir", default="3.ocr_output")
    parser.add_argument("--output", default="result_levenshtein.csv")
    args = parser.parse_args()

    print("=== Baseline: Levenshtein (CER + WER) ===")
    X_rows, labels = compute_features(Path(args.ocr_dir))

    import numpy as np
    X = np.array(X_rows)
    le = LabelEncoder()
    y = le.fit_transform(labels)

    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, C=1.0))
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(clf, X, y, cv=cv, scoring="accuracy")
    mean_acc, std_acc = float(scores.mean()), float(scores.std())

    print(f"Accuracy = {mean_acc:.4f} ± {std_acc:.4f} ({mean_acc*100:.2f}%)")

    pd.DataFrame([{
        "model": "Levenshtein (CER+WER)",
        "accuracy": round(mean_acc, 4),
        "std": round(std_acc, 4),
        "accuracy_pct": f"{mean_acc*100:.2f}%",
        "features": "CER, WER vs 1.original",
        "classifier": "Logistic Regression",
        "note": "Edit distance ký tự và từ"
    }]).to_csv(args.output, index=False)
    print(f"Đã ghi: {args.output}")


if __name__ == "__main__":
    main()