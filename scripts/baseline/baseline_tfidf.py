# baseline_tfidf.py
# =============================================
# Chạy: LOCAL (CPU)
# Cần : pip install scikit-learn
# Input: thư mục 3.ocr_output/
# Output: result_tfidf.csv
# =============================================
import argparse
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

CATEGORIES = ["1.original", "2.insert", "3.delete", "4.modify", "5.layout"]


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    return re.sub(r"\s+", " ", text).strip().lower()


def compute_features(ocr_dir: Path) -> tuple:
    original_dir = ocr_dir / "1.original"
    orig_texts = {}
    all_texts = []

    # Đọc tất cả text để fit vectorizer
    for txt_path in sorted(original_dir.rglob("*.txt")):
        rel = txt_path.relative_to(original_dir)
        text = normalize(txt_path.read_text(encoding="utf-8-sig", errors="ignore"))
        orig_texts[rel] = text
        all_texts.append(text)

    candidate_data = []
    for cat in ["2.insert", "3.delete", "4.modify", "5.layout"]:
        cat_dir = ocr_dir / cat
        if not cat_dir.exists():
            continue
        for txt_path in sorted(cat_dir.rglob("*.txt")):
            rel = txt_path.relative_to(cat_dir)
            if rel not in orig_texts:
                continue
            text = normalize(txt_path.read_text(encoding="utf-8-sig", errors="ignore"))
            all_texts.append(text)
            candidate_data.append((rel, cat, text))

    print(f"Fit TF-IDF vectorizer trên {len(all_texts)} văn bản ...")
    vectorizer = TfidfVectorizer(
        analyzer="char_wb",   # character n-gram phù hợp với tiếng Việt
        ngram_range=(2, 4),
        max_features=50000,
        sublinear_tf=True,
    )
    vectorizer.fit(all_texts)

    X_rows, labels = [], []

    # 1.original so với chính nó
    print(f"  [1.original] {len(orig_texts)} file")
    for rel, orig_text in orig_texts.items():
        v = vectorizer.transform([orig_text])
        sim = float(cosine_similarity(v, v)[0, 0])
        X_rows.append([sim])
        labels.append("1.original")

    # Nhóm tấn công so với original
    for rel, cat, cand_text in candidate_data:
        orig_text = orig_texts.get(rel)
        if not orig_text:
            continue
        v_orig = vectorizer.transform([orig_text])
        v_cand = vectorizer.transform([cand_text])
        sim = float(cosine_similarity(v_orig, v_cand)[0, 0])
        X_rows.append([sim])
        labels.append(cat)

    from collections import Counter
    print("  Phân bố nhãn:", Counter(labels))
    return X_rows, labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ocr-dir", default="3.ocr_output")
    parser.add_argument("--output", default="result_tfidf.csv")
    args = parser.parse_args()

    print("=== Baseline: TF-IDF Cosine Similarity ===")
    X_rows, labels = compute_features(Path(args.ocr_dir))

    X = np.array(X_rows)
    le = LabelEncoder()
    y = le.fit_transform(labels)

    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, C=1.0))
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(clf, X, y, cv=cv, scoring="accuracy")
    mean_acc, std_acc = float(scores.mean()), float(scores.std())

    print(f"Accuracy = {mean_acc:.4f} ± {std_acc:.4f} ({mean_acc*100:.2f}%)")

    pd.DataFrame([{
        "model": "TF-IDF cosine similarity",
        "accuracy": round(mean_acc, 4),
        "std": round(std_acc, 4),
        "accuracy_pct": f"{mean_acc*100:.2f}%",
        "features": "Char n-gram (2-4) TF-IDF cosine vs 1.original",
        "classifier": "Logistic Regression",
        "note": "Bag-of-words dựa trên n-gram ký tự"
    }]).to_csv(args.output, index=False)
    print(f"Đã ghi: {args.output}")


if __name__ == "__main__":
    main()