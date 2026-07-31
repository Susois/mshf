# MSHF — HƯỚNG DẪN CHẠY 27 YÊU CẦU

## Chuẩn bị

```powershell
git clone https://github.com/Susois/mshf.git
cd mshf
python -m pip install -r requirements.txt
python -m pytest tests -q
```

Dữ liệu không nằm trong GitHub. Đặt `mshf`, `Tuan1_2`, `Tuan5`, `Tuan6` chung một thư mục cha hoặc cấu hình biến môi trường theo `.env.example`.

---

## Giai đoạn 1 — Chuẩn hóa dữ liệu và baseline

### Yêu cầu 1 — Đối soát đúng 1.490 PDF

```powershell
python -m mshf.dataset_audit --strict --folds 5
```

Kết quả: `outputs/audit/audit_summary.json` phải có `source_documents=298`, `samples=1490`, mỗi category có 298 mẫu, `missing_files=0` và `ok=true`. Nếu sai, dừng pipeline và xử lý file thiếu/dư trước khi chạy tiếp.

### Yêu cầu 2 — Chuẩn hóa metadata của năm variants

```powershell
python -m mshf.dataset_audit --strict --folds 5
```

Kết quả: `outputs/audit/dataset_inventory.csv` chứa `sample_id`, `source_document_id`, `category`, `label`, `is_tampered` và đường dẫn PDF/OCR/GT/layout cho từng mẫu.

### Yêu cầu 3 — Kiểm tra missing, duplicate, OCR và feature

```powershell
python -m mshf.dataset_audit --strict --folds 5
python -m mshf.build_dataset --output outputs/enhanced_dataset.csv
```

Kết quả audit nằm trong `outputs/audit/audit_report.txt`. Dataset feature phải có 1.490 dòng, không có feature thiếu hoặc vô hạn. Lệnh `--strict` sẽ dừng khi input không hợp lệ.

### Yêu cầu 4 — Khóa 5-fold source-disjoint split

```powershell
python -m mshf.dataset_audit --strict --folds 5
```

Kết quả: `outputs/audit/source_splits.csv`. Mỗi `source_document_id` chỉ thuộc một fold. Giữ nguyên file này cho tất cả thí nghiệm.

### Yêu cầu 5 — Chạy 8 feature gốc và 40 feature MSHF

```powershell
python -m mshf.build_dataset --output outputs/enhanced_dataset.csv
python -m mshf.train --dataset outputs/enhanced_dataset.csv --splits outputs/audit/source_splits.csv --out outputs/training --task both --models single stacking two_stage
```

Kết quả: ablation `A` dùng 8 feature gốc; `A+B1+C` dùng 40 feature MSHF. Bảng so sánh nằm tại `outputs/training/comparison.csv`.

### Yêu cầu 6 — Báo cáo binary và five-class metrics

```powershell
python -m mshf.train --dataset outputs/enhanced_dataset.csv --splits outputs/audit/source_splits.csv --out outputs/training --task both --models single stacking two_stage
```

Kết quả: `outputs/training/training_summary.json` và `report.txt`. Binary dùng macro-F1, balanced accuracy, F1 original, MCC, AUROC và AUPRC; five-class dùng macro-F1, per-class F1, balanced accuracy, confusion matrix và macro AUROC.

---

## Giai đoạn 2 — Hoàn thiện phương pháp đánh giá

### Yêu cầu 7 — Nested out-of-fold stacking

```powershell
python -m mshf.train --dataset outputs/enhanced_dataset.csv --splits outputs/audit/source_splits.csv --out outputs/training --task both --models stacking
```

Kết quả: các file `outputs/training/predictions_*_stacking.csv`. Base learners tạo inner OOF probabilities trước khi meta-XGBoost được huấn luyện.

### Yêu cầu 8 — Class weighting, threshold validation và calibration

```powershell
python -m mshf.train --dataset outputs/enhanced_dataset.csv --splits outputs/audit/source_splits.csv --out outputs/training --task binary --models single stacking
```

Kết quả: threshold từng fold, Brier và ECE nằm trong `training_summary.json`. Threshold được chọn trong inner GroupKFold, không dùng outer test. **Hiện mã đã có threshold validation và calibration metrics; probability calibrator Platt/isotonic chưa được triển khai, vì vậy chưa được tuyên bố đã hoàn thành calibration model.**

### Yêu cầu 9 — Bootstrap CI theo source document

```powershell
python -m mshf.train --dataset outputs/enhanced_dataset.csv --splits outputs/audit/source_splits.csv --out outputs/training --task both
```

Kết quả: `accuracy_ci95` và `macro_f1_ci95` trong `training_summary.json`, được bootstrap theo `source_document_id`.

### Yêu cầu 10 — Paired statistical tests và effect size

```powershell
python -m mshf.train --dataset outputs/enhanced_dataset.csv --splits outputs/audit/source_splits.csv --out outputs/training --task both
```

Kết quả: `outputs/training/statistical_tests.json`, gồm Wilcoxon p-value, paired difference và Cohen’s dz giữa baseline A và full MSHF.

### Yêu cầu 11 — Lưu checkpoint, config, seed, split và prediction

```powershell
python -m mshf.train --dataset outputs/enhanced_dataset.csv --splits outputs/audit/source_splits.csv --out outputs/training --task both --models single stacking two_stage
```

Kết quả: `mshf_*.joblib`, `run_config.json`, `predictions_*.csv`, `training_summary.json`; split nằm tại `outputs/audit/source_splits.csv`.

---

## Giai đoạn 3 — Hoàn thiện ground truth

### Yêu cầu 12 — Chuẩn hóa attack manifest

```powershell
python -m mshf.manifest --source ..\Tuan1_2\VEDTD\1.pdfs\manifest.csv --output outputs\manifest\attack_manifest.csv
```

Kết quả: manifest chuẩn gồm ID, attack type, subtype, severity, page, line, token, bbox và nguồn annotation. Trường chưa có trong manifest gốc được ghi `unknown` hoặc rỗng.

### Yêu cầu 13 — Đa dạng hóa vị trí, subtype và severity

```powershell
python -m mshf.manifest --source ..\Tuan1_2\VEDTD\1.pdfs\manifest.csv --output outputs\manifest\attack_manifest.csv
```

Kết quả hiện tại chỉ chuẩn hóa metadata. **Việc sinh lại PDF với vị trí/subtype/severity đa dạng phải được bổ sung tại bộ sinh tấn công nguồn; repository hiện chưa có lệnh tự động tạo các PDF mới.** Sau khi sinh lại, chạy lệnh trên để chuẩn hóa manifest.

### Yêu cầu 14 — Tạo authentic controls

```powershell
python -m mshf.create_controls --out-dir outputs/controls
# test nhanh voi 5 documents dau:
python -m mshf.create_controls --out-dir outputs/controls --max-docs 5
```

Kết quả: JPEG recompression, blur, resize, skew, contrast, noise và perspective (mỗi loại 5 levels) từ original, lưu dưới `outputs/controls/<corruption>/<level>/` cùng `control_manifest.csv` chứa provenance (source/output path + SHA256 hash), parameters và seed. Không sửa PDF nguồn. Cần cài `PyMuPDF` để render PDF sang ảnh.

### Yêu cầu 15 — Ánh xạ manifest sang page, line, token và bbox

```powershell
python -m mshf.manifest --source ..\Tuan1_2\VEDTD\1.pdfs\manifest.csv --output outputs\manifest\attack_manifest.csv
python -m mshf.localize --out outputs\localization
```

Kết quả line-level nằm trong `outputs/localization/localization_details.json`. **BBox/token ground truth chỉ hợp lệ khi manifest nguồn chứa annotation trực tiếp; fallback OCR/GT không được gọi là bbox ground truth.**

### Yêu cầu 16 — Đánh giá lại localization

```powershell
python -m mshf.localize --out outputs\localization
```

Kết quả: `localization_metrics.csv` và `localization_details.json`. Hiện chạy line-level P/R/F1; region IoU/Dice cần bbox thật trong manifest.

---

## Giai đoạn 4 — Semantic Critical Change (B2)

### Yêu cầu 17 — Tính contradiction/entailment

Chuẩn bị line pairs:

```powershell
python -m mshf.semantic_features prepare --output outputs\semantic\line_pairs.jsonl
```

Chạy trên Colab GPU:

```bash
python -m mshf.colab_nli --pairs outputs/semantic/line_pairs.jsonl --output outputs/semantic/nli_scores.csv --model joeddav/xlm-roberta-large-xnli --batch-size 16
```

Kết quả: `nli_scores.csv` chứa contradiction, entailment và neutral cho từng `pair_id`.

### Yêu cầu 18 — Trích xuất thực thể, số liệu, ngày và đơn vị

```powershell
python -m mshf.semantic_features aggregate --pairs outputs\semantic\line_pairs.jsonl --nli outputs\semantic\nli_scores.csv --output outputs\semantic\b2_features.csv
```

Kết quả: các cột entity, numeric và unit change trong `b2_features.csv`. Entity hiện dùng heuristic; chưa phải Vietnamese NER chuyên biệt.

### Yêu cầu 19 — Trích xuất phủ định, nghĩa vụ, quyền hạn và logic

```powershell
python -m mshf.semantic_features aggregate --pairs outputs\semantic\line_pairs.jsonl --nli outputs\semantic\nli_scores.csv --output outputs\semantic\b2_features.csv
```

Kết quả: negation, obligation và logic change features trong `b2_features.csv`.

### Yêu cầu 20 — Tổng hợp feature có trọng số OCR confidence

```powershell
python -m mshf.semantic_features aggregate --pairs outputs\semantic\line_pairs.jsonl --nli outputs\semantic\nli_scores.csv --output outputs\semantic\b2_features.csv
```

Kết quả: contradiction mean/max/p90/high-count và critical-change features được tổng hợp theo từng `source_document_id + category`.

### Yêu cầu 21 — Ablation B2 và modify/low-severity

```powershell
python -m mshf.join_b2
python -m mshf.train --dataset outputs/enhanced_dataset_b2.csv --splits outputs/audit/source_splits.csv --out outputs/training_b2 --task both --models single stacking
python -m mshf.analyze_severity --predictions outputs/training_b2/predictions_is_tampered_A_B1_B2_C_stacking.csv
```

Kết quả: `join_b2` join `enhanced_dataset.csv` với `b2_features.csv` theo `source_document_id + category` (one-to-one, fill 0 cho original) và lưu `outputs/enhanced_dataset_b2.csv`. `train.py` đã đăng ký B2 columns, chạy đủ ablation `A`, `A+B1`, `A+B1+C`, `A+B1+B2`, `A+B1+B2+C` trên cùng fixed splits và lưu `b2_ablation_metrics.csv` + `run_config_b2.json`. Phân tích low-severity (`severity_metrics.csv`) chỉ có ý nghĩa khi manifest có severity thật.

---

## Giai đoạn 5 — Thực nghiệm hoàn chỉnh

### Yêu cầu 22 — Robustness matrix

```powershell
python -m mshf.create_controls --out-dir outputs/controls
python -m mshf.extract_control_features --dataset outputs/enhanced_dataset.csv --out-dir outputs/robustness
python -m mshf.robustness_eval --dataset outputs/enhanced_dataset.csv --splits outputs/audit/source_splits.csv --out-dir outputs/robustness
```

Kết quả: `outputs/controls/control_manifest.csv` (7 corruptions × 5 levels, kèm parameters/seed/hash provenance); `outputs/robustness/robustness_matrix.csv` với `corruption, level, macro_f1, balanced_accuracy, auroc, relative_drop_f1, sample_count`; predictions OOF từng corruption tại `predictions_<corruption>_<level>.csv`. Đánh giá dùng cùng fixed splits: train trên clean features, predict trên corrupted features. **Lưu ý: `extract_control_features` mô phỏng ảnh hưởng corruption bằng feature perturbation; chưa chạy lại OCR/layout thật trên từng control PDF — không tuyên bố tương đương pipeline OCR thật.**

### Yêu cầu 23 — Severity analysis

```powershell
python -m mshf.analyze_severity --predictions outputs/training/predictions_is_tampered_A_B1_C_stacking.csv --manifest outputs/manifest/attack_manifest.csv --out-dir outputs/analysis
```

Kết quả: `severity_metrics.csv`, `severity_predictions_joined.csv` và `severity_run_config.json`, group theo `attack_type + severity` với sample count. **Chỉ báo cáo khi manifest có severity thật (low/medium/high); script sẽ dừng nếu chỉ có `unknown`/rỗng.**

### Yêu cầu 24 — Unseen-subtype/generator evaluation

```powershell
python -m mshf.unseen_eval --dataset outputs/enhanced_dataset.csv --manifest outputs/manifest/attack_manifest.csv --out-dir outputs/unseen --target is_tampered
```

Kết quả: leave-one-subtype-out / leave-one-generator-out. Mỗi run giữ một subtype/generator hoàn toàn ngoài training và loại cả source document rò rỉ. Lưu `unseen_subtype_metrics.csv`, `unseen_generator_metrics.csv`, `predictions_holdout_<group>_<value>.csv` và `splits_holdout_<group>_<value>.json`. **Chỉ chạy khi manifest có `attack_subtype`/`generator_id` thật với ít nhất 2 nhóm.**

### Yêu cầu 25 — Cross-template evaluation

```powershell
python -m mshf.cross_template_eval --dataset outputs/enhanced_dataset.csv --manifest outputs/manifest/attack_manifest.csv --out-dir outputs/cross_template --target is_tampered
```

Kết quả: `LeaveOneGroupOut` theo `template_id`, toàn bộ source cùng template nằm cùng phía train/test và source document không rò rỉ. Lưu `cross_template_metrics.csv` (mean/std + sample count), `predictions_fold_*.csv`, `splits_fold_*.json` và `run_config.json`. **Chỉ chạy khi có `template_id` thật đã xác minh với ít nhất 2 templates.**

### Yêu cầu 26 — Case studies, calibration plot và failure taxonomy

```powershell
python -m mshf.create_report --predictions outputs/training/predictions_is_tampered_A_B1_C_stacking.csv --localization outputs/localization/localization_details.json --out-dir outputs/report
```

Kết quả: `case_studies.csv` (TP/FP/FN/TN đại diện), `calibration_plot.png` + `calibration_bins.csv` (reliability diagram từ OOF probability, kèm Brier/ECE), `failure_taxonomy.csv` (phân loại lỗi kèm possible cause và link localization), `report_summary.md` và `run_config.json`. **Failure taxonomy hiện dùng heuristic từ confidence + localization; chưa phải annotation chuyên sâu cho OCR/alignment/semantic/geometry failure.**

### Yêu cầu 27 — Chuẩn hóa mã nguồn và tái lập

```powershell
python -m pytest tests -q
python -m mshf.dataset_audit --strict --folds 5
```

Kết quả: tests phải pass; audit phải xác nhận 298 source/1.490 mẫu. Lưu cùng `requirements.txt`, `.env.example`, fixed splits, `run_config.json`, prediction per sample, model/checkpoint revision và lệnh chạy.

---

## Trạng thái thực tế

| Phạm vi | Trạng thái |
|---|---|
| Yêu cầu 1–7, 9–12, 14, 16–22, 26, 27 | Có lệnh/module để chạy |
| Yêu cầu 8 | Có threshold và calibration metrics; chưa có Platt/isotonic calibrator |
| Yêu cầu 13, 15 | Có schema/fallback; cần annotation hoặc generator mở rộng |
| Yêu cầu 22 | Có runner; feature control dùng perturbation, chưa chạy lại OCR thật |
| Yêu cầu 23–25 | Có runner; chỉ có ý nghĩa khi manifest có severity/subtype/generator/template thật |

Không sử dụng dòng mô tả “chưa có lệnh” như bằng chứng đã hoàn thành. Chỉ đưa một yêu cầu vào bài báo sau khi đã có artifact đầu ra và kiểm tra trên full 298 source.
