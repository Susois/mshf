# baseline_sbert.py
# =============================================
# Chạy: COLAB (GPU) ← khác với các file còn lại
# Cần : pip install sentence-transformers scikit-learn
# Input: thư mục 3.ocr_output/
# Output: result_sbert.csv
#
# Lý do cần GPU: encode ~1490 văn bản bằng SBERT trên CPU rất chậm
# Trên GPU T4 Colab ước tính ~5-10 phút, CPU có thể >1 giờ
# =============================================
import argparse
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer, util
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

CATEGORIES = ["1.original", "2.insert", "3.delete", "4.modify", "5.layout"]
# Model đa ngôn ngữ — hỗ trợ tiếng Việt, download ~420MB lần đầu
SBERT_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    return re.sub(r"\s+", " ", text).strip().lower()


def chunk_and_embed(text: str, model: SentenceTransformer, max_words: int = 200):
    words = text.split()
    chunks = [" ".join(words[i:i+max_words]) for i in range(0, len(words), max_words)]
    if not chunks:
        chunks = [""]
    embeddings = model.encode(chunks, convert_to_tensor=True, show_progress_bar=False)
    return embeddings.mean(dim=0)


def compute_features(ocr_dir: Path, device: str) -> tuple:
    print(f"Loading SBERT: {SBERT_MODEL} on {device} ...")
    sbert = SentenceTransformer(SBERT_MODEL, device=device)

    original_dir = ocr_dir / "1.original"
    orig_embs = {}

    print("Encode bản gốc (1.original) ...")
    orig_files = sorted(original_dir.rglob("*.txt"))
    for i, txt_path in enumerate(orig_files):
        rel = txt_path.relative_to(original_dir)
        text = normalize(txt_path.read_text(encoding="utf-8-sig", errors="ignore"))
        orig_embs[rel] = chunk_and_embed(text, sbert)
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(orig_files)}")

    X_rows, labels = [], []

    # 1.original so với chính nó → similarity = 1.0
    for rel in orig_embs:
        X_rows.append([1.0])
        labels.append("1.original")

    # Các nhóm tấn công
    for cat in ["2.insert", "3.delete", "4.modify", "5.layout"]:
        cat_dir = ocr_dir / cat
        if not cat_dir.exists():
            continue
        cat_files = sorted(cat_dir.rglob("*.txt"))
        print(f"Encode [{cat}] {len(cat_files)} file ...")
        for i, txt_path in enumerate(cat_files):
            rel = txt_path.relative_to(cat_dir)
            if rel not in orig_embs:
                continue
            text = normalize(txt_path.read_text(encoding="utf-8-sig", errors="ignore"))
            cand_emb = chunk_and_embed(text, sbert)
            sim = float(util.cos_sim(orig_embs[rel], cand_emb)[0][0])
            X_rows.append([sim])
            labels.append(cat)
            if (i + 1) % 50 == 0:
                print(f"  {i+1}/{len(cat_files)}")

    return X_rows, labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ocr-dir", default="3.ocr_output")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", default="result_sbert.csv")
    args = parser.parse_args()

    print("=== Baseline: SBERT Multilingual Cosine Similarity ===")
    X_rows, labels = compute_features(Path(args.ocr_dir), args.device)

    X = np.array(X_rows)
    le = LabelEncoder()
    y = le.fit_transform(labels)

    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, C=1.0))
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(clf, X, y, cv=cv, scoring="accuracy")
    mean_acc, std_acc = float(scores.mean()), float(scores.std())

    print(f"Accuracy = {mean_acc:.4f} ± {std_acc:.4f} ({mean_acc*100:.2f}%)")

    pd.DataFrame([{
        "model": "SBERT (multilingual)",
        "accuracy": round(mean_acc, 4),
        "std": round(std_acc, 4),
        "accuracy_pct": f"{mean_acc*100:.2f}%",
        "features": "SBERT (paraphrase-multilingual-MiniLM-L12-v2) cosine vs 1.original",
        "classifier": "Logistic Regression",
        "note": "Sentence embedding đa ngôn ngữ"
    }]).to_csv(args.output, index=False)
    print(f"Đã ghi: {args.output}")


if __name__ == "__main__":
    main()