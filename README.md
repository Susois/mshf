# MSHF — Multi-Scale Hybrid Fusion

MSHF mở rộng một hệ thống phát hiện giả mạo tài liệu có tham chiếu: giữ 8 feature OCR–PhoBERT–LayoutLMv3 và XGBoost, bổ sung line evidence (B1), geometry evidence (C), Semantic Critical Change (B2), localization và protocol đánh giá source-disjoint.

## 1. Dữ liệu chuẩn

Dữ liệu không nằm trong repository. `config.py` đọc đường dẫn từ biến môi trường `MSHF_PROJECT_ROOT` hoặc các biến override trong `.env.example`:

| Nguồn | Đường dẫn | Quy mô chuẩn |
|---|---|---:|
| PDF | `Tuan1_2/VEDTD/1.pdfs/{1.original..5.layout}` | 298 × 5 = 1.490 |
| Ground truth | `Tuan1_2/VEDTD/2.ground_truth` | 298 × 5 |
| OCR | `Tuan1_2/VEDTD/3.ocr_output` | 298 × 5 |
| Layout JSON | `Tuan5/4.layout_ocr` | 298 × 5 |
| 8 feature gốc | `Tuan6/src/hybrid_fusion_dataset.csv` | 1.490 dòng |

Mỗi source có `1 original + 4 attack = 5 variants`. Tất cả variants của cùng `source_document_id` luôn ở cùng fold.

## 2. Kiến trúc feature

- **A (8):** CER, WER, 5 thống kê PhoBERT, LayoutLMv3 cosine.
- **B1 (16):** line insert/delete/modify, operation ratios, line CER và semantic risk dựa luật.
- **C (16):** geometry delta, robust registration và bbox residual.
- **B2 (tùy chọn GPU):** contradiction/entailment, entity, numerical, unit, negation, obligation và logic changes.

Pipeline CPU chuẩn sử dụng 40 feature A+B1+C. B2 chỉ được đưa vào ablation sau khi file NLI đầy đủ được tạo; không điền 0 giả khi thiếu artifact.

## 3. Cấu trúc dự án

```text
mshf/
├── config.py
├── requirements.txt
├── README.md
├── COLAB_GUIDE.md
├── mshf/
│   ├── dataset_audit.py       # đối soát dữ liệu + fixed splits
│   ├── build_dataset.py       # build 40 feature
│   ├── line_align.py
│   ├── line_features.py       # B1
│   ├── geometric_features.py  # C
│   ├── semantic_risk.py
│   ├── semantic_features.py   # chuẩn bị/aggregate B2
│   ├── colab_nli.py           # inference NLI trên GPU
│   ├── manifest.py            # manifest chuẩn
│   ├── models.py              # single/OOF stacking/two-stage
│   ├── evaluate.py
│   ├── train.py
│   ├── localization_gt.py
│   └── localize.py
├── tests/
└── outputs/                   # artifact tái tạo, không phải mã nguồn
```

## 4. Cài đặt

```powershell
git clone https://github.com/Susois/mshf.git mshf
cd mshf
python -m pip install -r requirements.txt
python -m pytest tests -q
```

## 5. Trình tự chạy bắt buộc

### Giai đoạn 1 — Audit, metadata, split và baseline

```powershell
python -m mshf.dataset_audit --strict --folds 5
python -m mshf.build_dataset
python -m mshf.train --task both --models single stacking two_stage
```

Artifact:

- `outputs/audit/dataset_inventory.csv`
- `outputs/audit/audit_summary.json`
- `outputs/audit/source_splits.csv`
- `outputs/enhanced_dataset.csv`
- `outputs/training/comparison.csv`
- `outputs/training/predictions_*.csv`
- `outputs/training/training_summary.json`

### Giai đoạn 2 — Đánh giá chặt chẽ

`train.py` sử dụng fixed source splits, OOF stacking, inner-fold threshold validation, cluster bootstrap theo source, paired Wilcoxon/effect size và lưu checkpoint/config/predictions.

Metric binary: balanced accuracy, macro-F1, original-class F1, MCC, AUROC, AUPRC, Brier, ECE. Accuracy chỉ là metric phụ vì tỷ lệ original:tampered = 1:4.

Metric five-class: macro-F1, weighted-F1, per-class precision/recall/F1, balanced accuracy, confusion matrix và macro OVR AUROC.

### Giai đoạn 3 — Manifest và localization

```powershell
python -m mshf.manifest
python -m mshf.localize
```

Manifest cũ được chuẩn hóa tại `outputs/manifest/attack_manifest.csv`. Trường chưa có bằng chứng trực tiếp được ghi `unknown`/rỗng, không được coi là annotation chắc chắn. Localization hiện hỗ trợ GT text fallback; bbox/region metrics chỉ được báo cáo sau khi manifest có bbox thật.

### Giai đoạn 4 — Semantic Critical Change B2

Chuẩn bị toàn bộ cặp dòng:

```powershell
python -m mshf.semantic_features prepare --output outputs/semantic/line_pairs.jsonl
```

Chạy `mshf.colab_nli` trên Colab theo `COLAB_GUIDE.md`, tải `nli_scores.csv` về rồi aggregate:

```powershell
python -m mshf.semantic_features aggregate `
  --pairs outputs/semantic/line_pairs.jsonl `
  --nli outputs/semantic/nli_scores.csv `
  --output outputs/semantic/b2_features.csv
```

Sau đó join B2 theo `source_document_id + category` để chạy ablation `A+B1+B2+C`. Phân tích riêng lớp `modify` và severity thấp chỉ thực hiện khi metadata severity tồn tại.

### Giai đoạn 5 — Thực nghiệm hoàn chỉnh

Các experiment phải dùng đúng `outputs/audit/source_splits.csv`:

- robustness matrix: JPEG, blur, resize, skew, contrast/noise/perspective;
- severity analysis;
- unseen subtype/generator;
- cross-template khi có `template_id`;
- calibration curve, case studies và failure taxonomy.

Nếu severity/subtype/template chưa được annotation, báo `not_available`; không sinh số liệu giả.

## 6. Ánh xạ 27 yêu cầu sang artifact

| # | Yêu cầu | Lệnh/module | Artifact/chứng cứ |
|---:|---|---|---|
| 1–3 | 1.490 mẫu, metadata, missing/duplicate/OCR/feature QC | `dataset_audit` | `outputs/audit/*` |
| 4 | Khóa 5-fold source-disjoint | `dataset_audit --folds 5` | `source_splits.csv` |
| 5 | 8 và 40 feature | `build_dataset`, ablation A/full | `enhanced_dataset.csv` |
| 6 | Binary/five-class metrics | `train --task both` | summary/comparison |
| 7 | Nested OOF stacking | `models.StackingModel.fit_oof` | predictions stacking |
| 8 | Weight/threshold/calibration | inner group validation | fold thresholds, Brier/ECE |
| 9 | Cluster bootstrap | `evaluate.cluster_bootstrap_ci` | CI trong JSON |
| 10 | Paired test/effect size | `evaluate.paired_tests` | `statistical_tests.json` |
| 11 | Checkpoint/config/split/prediction | `train` | `outputs/training/*` |
| 12,15–16 | Manifest và localization | `manifest`, `localize` | manifest/localization |
| 13 | Subtype/severity | manifest schema + attack generator nguồn | metadata đã kiểm chứng |
| 14 | Authentic controls | thực nghiệm robustness có provenance | controls + run config |
| 17–20 | NLI/entity/numeric/logic/confidence | `semantic_features`, `colab_nli` | B2 feature CSV |
| 21 | Ablation B2/modify/low | join B2 + grouped analysis | comparison/slices |
| 22 | Robustness | experiment theo corruption | robustness matrix |
| 23 | Severity | slice theo severity | severity results |
| 24 | Unseen generator | holdout subtype | unseen results |
| 25 | Cross-template | group theo template_id | template results |
| 26 | Cases/calibration/failure | OOF predictions/details | figure/source tables |
| 27 | Tái lập | README, Colab, run config, tests | toàn bộ repo/artifacts |

## 7. Nguyên tắc khoa học

1. Không random split từng PDF; split theo 298 source.
2. Không tune threshold/calibration/hyperparameter trên outer test.
3. Không dùng GT text làm input inference; GT/manifest chỉ dùng đánh giá.
4. Bootstrap và statistical tests theo source, không coi 1.490 variants là độc lập.
5. Không dùng accuracy 80% làm bằng chứng binary vì majority baseline đã đạt 80%.
6. Smoke test không phải kết quả bài báo.
7. Ghi model revision, package versions, seed, split IDs và command chạy.
8. Không công bố PDF nếu chưa xác minh quyền dữ liệu.

## 9. Repository và dữ liệu

Repository chỉ chứa mã nguồn và tài liệu kỹ thuật. Raw/tampered PDFs, OCR/GT text, manifests có nội dung thật, outputs, checkpoints và bản thảo nội bộ bị loại bởi `.gitignore`.

- Cấu hình đường dẫn bằng các biến môi trường trong `.env.example`.
- Xem `DATA_AVAILABILITY.md` trước khi chia sẻ dữ liệu.
- Xem `THIRD_PARTY_LICENSES.md` trước khi chọn hoặc phân phối checkpoint.
- Xem `RELEASE_CHECKLIST.md` trước khi push hoặc đổi repository sang public.
- Trong giai đoạn chưa công bố, nên giữ GitHub repository ở chế độ **Private**.

Chưa gắn giấy phép mã nguồn cho đến khi quyền sở hữu được xác nhận với tác giả/nhóm/cơ sở liên quan. Việc không có `LICENSE` có nghĩa là không mặc nhiên cấp quyền sao chép, sửa đổi hoặc phân phối.

## 10. Kiểm tra nhanh

```powershell
python -m pytest tests -q
python -m mshf.dataset_audit --strict
python -m mshf.build_dataset --max-per-cat 5 --output outputs/smoke/enhanced.csv
python -m mshf.train --dataset outputs/smoke/enhanced.csv --splits outputs/smoke/no_fixed_split.csv --out outputs/smoke/training --folds 3 --task both --models single
```

Kết quả smoke chỉ xác nhận pipeline chạy được. Bảng bài báo phải lấy từ full 298 source với fixed splits.
