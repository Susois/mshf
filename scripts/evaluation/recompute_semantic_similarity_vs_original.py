# recompute_semantic_similarity_vs_original.py
from __future__ import annotations

import argparse
import csv
import re
import sys
import unicodedata
from pathlib import Path
from statistics import mean

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

CATEGORIES = ["1.original", "2.insert", "3.delete", "4.modify", "5.layout"]


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8-sig", errors="ignore")


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def chunk_text(text: str, max_words: int = 200) -> list[str]:
    words = text.split()
    return [" ".join(words[i:i + max_words]) for i in range(0, len(words), max_words)] or [""]


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    summed = torch.sum(token_embeddings * mask, dim=1)
    counts = torch.clamp(mask.sum(dim=1), min=1e-9)
    return summed / counts


def embed_chunks(chunks: list[str], tokenizer, model, device, batch_size: int = 8) -> np.ndarray:
    embeddings = []
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        inputs = tokenizer(batch, padding=True, truncation=True, max_length=256, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        emb = mean_pooling(outputs, inputs["attention_mask"])
        emb = torch.nn.functional.normalize(emb, p=2, dim=1)
        embeddings.append(emb.cpu().numpy())
    return np.concatenate(embeddings, axis=0)


def cosine_sim_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a @ b.T  # đã normalize nên dot product = cosine similarity


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Tính lại semantic similarity: so OCR output của MỌI category với ground truth của 1.original."
    )
    parser.add_argument("--root", default=".")
    parser.add_argument("--gt-dir", default="2.ground_truth")
    parser.add_argument("--ocr-dir", default="3.ocr_output")
    parser.add_argument("--report", default="semantic_similarity_vs_original.csv")
    parser.add_argument("--model-name", default="vinai/phobert-base-v2")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-words-per-chunk", type=int, default=200)
    args = parser.parse_args()

    root = Path(args.root).resolve()
    gt_root = root / args.gt_dir
    ocr_root = root / args.ocr_dir
    report_path = root / args.report
    original_gt_dir = gt_root / "1.original"

    print(f"Loading {args.model_name} on {args.device} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModel.from_pretrained(args.model_name).to(args.device)
    model.eval()

    rows = []
    for category in CATEGORIES:
        ocr_cat_dir = ocr_root / category
        if not ocr_cat_dir.exists():
            print(f"[WARN] Bỏ qua, không thấy: {ocr_cat_dir}")
            continue

        txt_files = sorted(ocr_cat_dir.rglob("*.txt"))
        print(f"\n[{category}] {len(txt_files)} file")

        for idx, ocr_txt_path in enumerate(txt_files, 1):
            rel = ocr_txt_path.relative_to(ocr_cat_dir)
            gt_path = original_gt_dir / rel

            row = {
                "category": category, "file": str(rel).replace("\\", "/"),
                "status": "", "error": "",
                "mean_similarity": "", "min_similarity": "", "max_similarity": "", "std_similarity": "",
                "ref_to_hyp_mean": "", "hyp_to_ref_mean": "",
                "num_chunks_ref": "", "num_chunks_hyp": "",
            }

            try:
                if not gt_path.exists():
                    row["status"] = "missing_original_ground_truth"
                    rows.append(row)
                    continue

                ref_text = normalize_text(read_text(gt_path))      # ground truth của 1.original
                hyp_text = normalize_text(read_text(ocr_txt_path)) # OCR output của category hiện tại

                ref_chunks = chunk_text(ref_text, args.max_words_per_chunk)
                hyp_chunks = chunk_text(hyp_text, args.max_words_per_chunk)

                ref_emb = embed_chunks(ref_chunks, tokenizer, model, args.device)
                hyp_emb = embed_chunks(hyp_chunks, tokenizer, model, args.device)

                sim_matrix = cosine_sim_matrix(ref_emb, hyp_emb)  # shape (n_ref, n_hyp)

                ref_to_hyp_mean = float(sim_matrix.max(axis=1).mean())  # mỗi chunk ref khớp tốt nhất với chunk hyp nào
                hyp_to_ref_mean = float(sim_matrix.max(axis=0).mean())

                row.update({
                    "mean_similarity": f"{sim_matrix.mean():.6f}",
                    "min_similarity": f"{sim_matrix.min():.6f}",
                    "max_similarity": f"{sim_matrix.max():.6f}",
                    "std_similarity": f"{sim_matrix.std():.6f}",
                    "ref_to_hyp_mean": f"{ref_to_hyp_mean:.6f}",
                    "hyp_to_ref_mean": f"{hyp_to_ref_mean:.6f}",
                    "num_chunks_ref": len(ref_chunks),
                    "num_chunks_hyp": len(hyp_chunks),
                    "status": "ok",
                })
                print(f"  [{idx}/{len(txt_files)}] OK: {rel}  mean_sim={sim_matrix.mean():.4f}")

            except Exception as e:
                row["status"] = "error"
                row["error"] = repr(e)
                print(f"  [ERROR] {rel}: {repr(e)}", file=sys.stderr)

            rows.append(row)

    fieldnames = list(rows[0].keys())
    with report_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nĐã ghi: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())