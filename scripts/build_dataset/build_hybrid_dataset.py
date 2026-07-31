import pandas as pd

def strip_ext(filename: str) -> str:
    return filename.rsplit(".", 1)[0]

# --- Đọc 3 báo cáo ---
cer_wer = pd.read_csv("ocr_eval_report_vs_original.csv", encoding="utf-8-sig")
semantic = pd.read_csv("semantic_similarity_vs_original.csv", encoding="utf-8-sig")
layout = pd.read_csv("layout_embedding_similarity_report.csv", encoding="utf-8-sig")

# --- Chuẩn hóa khóa join: bỏ phần đuôi file (.txt / .npy), đổi tên cột cho khớp ---
cer_wer["key"] = cer_wer["file"].apply(strip_ext)
semantic["key"] = semantic["file"].apply(strip_ext)
layout["key"] = layout["filename"].apply(strip_ext)
layout = layout.rename(columns={"attack_group": "category"})

cer_wer = cer_wer[cer_wer["status"] == "ok"][["category", "key", "cer", "wer"]]
semantic = semantic[semantic["status"] == "ok"][
    ["category", "key", "mean_similarity", "min_similarity", "std_similarity",
     "ref_to_hyp_mean", "hyp_to_ref_mean"]
]
layout_attacks = layout[["category", "key", "layoutlmv3_cosine_similarity"]]

# --- Gộp CER/WER + semantic (cả 2 đều có đủ 5 category, bao gồm 1.original) ---
merged = cer_wer.merge(semantic, on=["category", "key"], how="inner")

# --- Gộp layout: chỉ có 4 nhóm tấn công, KHÔNG có 1.original ---
merged = merged.merge(layout_attacks, on=["category", "key"], how="left")

# --- Với 1.original: tự điền layout similarity = 1.0 (so với chính nó là hoàn hảo) ---
merged.loc[merged["category"] == "1.original", "layoutlmv3_cosine_similarity"] = 1.0

merged = merged.rename(columns={"category": "label"})
merged.to_csv("hybrid_fusion_dataset.csv", index=False)

print(merged["label"].value_counts())
print(merged.isna().sum())