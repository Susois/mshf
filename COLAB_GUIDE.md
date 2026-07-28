# Google Colab Guide — Hoàn thiện MSHF theo 5 giai đoạn

Hướng dẫn này chạy tuần tự từ audit 1.490 PDF đến bộ artifact dùng cho báo cáo. Không dùng kết quả subset/smoke làm kết quả bài báo.

## 0. Chuẩn bị Colab

Bật **Runtime → Change runtime type → T4 GPU**, sau đó mount Drive:

```python
from google.colab import drive
drive.mount('/content/drive')
```

Đặt dự án và dữ liệu theo cấu trúc:

```text
/content/drive/MyDrive/mshf_workspace/
├── mshf/
├── Tuan1_2/VEDTD/
├── Tuan5/4.layout_ocr/
└── Tuan6/src/hybrid_fusion_dataset.csv
```

```python
%cd /content/drive/MyDrive/mshf_workspace/mshf
!python -m pip install -q -r requirements.txt
!python -m pip install -q torch transformers sentencepiece accelerate
!nvidia-smi
```

> `config.py` mặc định tìm thư mục cha của `mshf`, nên cấu trúc trên phải được giữ nguyên.

---

# GIAI ĐOẠN 1 — CHUẨN HÓA DỮ LIỆU VÀ BASELINE

## 1.1. Đối soát 1.490 PDF và chất lượng input

```python
!python -m mshf.dataset_audit --strict --folds 5
```

Kết quả hợp lệ phải có:

```text
source_documents = 298
samples = 1490
mỗi category = 298
missing_files = 0
empty_ocr = 0
ok = true
```

Đọc báo cáo:

```python
import json, pandas as pd
print(json.load(open('outputs/audit/audit_summary.json', encoding='utf-8')))
display(pd.read_csv('outputs/audit/dataset_inventory.csv').head())
display(pd.read_csv('outputs/audit/source_splits.csv').groupby('fold').size())
```

Nếu strict audit thất bại, dừng pipeline và xử lý file thiếu/ngoài cấu trúc. Không tự điền feature 0 cho input bị mất.

## 1.2. Khóa split

`outputs/audit/source_splits.csv` là split chính thức. Sao lưu file này cùng bài báo:

```python
!cp outputs/audit/source_splits.csv /content/drive/MyDrive/mshf_workspace/mshf/source_splits_locked.csv
```

Không tái tạo split giữa các baseline. Mọi variants của cùng source phải có cùng fold.

## 1.3. Build 40 feature A+B1+C

```python
!python -m mshf.build_dataset --output outputs/enhanced_dataset.csv
```

Kiểm tra:

```python
df = pd.read_csv('outputs/enhanced_dataset.csv')
print(df.shape)                         # kỳ vọng (1490, 45): 5 metadata + 40 feature
print(df.source_document_id.nunique()) # 298
print(df.label.value_counts())         # 298 mỗi lớp
```

## 1.4. Baseline 8 feature và MSHF 40 feature

```python
!python -m mshf.train \
  --dataset outputs/enhanced_dataset.csv \
  --splits outputs/audit/source_splits.csv \
  --out outputs/training \
  --task both \
  --models single stacking two_stage
```

Artifact chính:

```python
display(pd.read_csv('outputs/training/comparison.csv'))
print(open('outputs/training/report.txt', encoding='utf-8').read())
```

Binary task có tỷ lệ 1:4. Dùng macro-F1, balanced accuracy, F1 original, MCC, AUROC và AUPRC; accuracy không phải metric chính.

---

# GIAI ĐOẠN 2 — HOÀN THIỆN PHƯƠNG PHÁP ĐÁNH GIÁ

## 2.1. Nested out-of-fold stacking

Lệnh ở mục 1.4 tự động gọi `StackingModel.fit_oof` trong từng outer fold. Base learner tạo inner OOF probabilities trước khi meta-XGBoost được fit. Outer test không được dùng cho meta training.

Kiểm tra prediction riêng:

```python
from pathlib import Path
for p in sorted(Path('outputs/training').glob('predictions_*stacking.csv')):
    print(p, pd.read_csv(p).shape)
```

## 2.2. Threshold validation và calibration metrics

Binary threshold được chọn bằng inner GroupKFold theo macro-F1. Threshold từng fold nằm trong `training_summary.json`:

```python
summary = json.load(open('outputs/training/training_summary.json', encoding='utf-8'))
for run in summary:
    if run['target'] == 'is_tampered':
        print(run['name'], [(x['fold'], x['threshold']) for x in run['folds']])
```

Brier và ECE được tính trên OOF probability. Không chọn threshold trên outer test.

## 2.3. Cluster bootstrap và paired statistics

Khoảng tin cậy bootstrap resample theo 298 `source_document_id`, không theo 1.490 rows độc lập.

```python
print(json.dumps(json.load(open('outputs/training/statistical_tests.json', encoding='utf-8')), indent=2))
```

Kết quả gồm paired difference, Wilcoxon p-value và Cohen's dz. Báo effect size cùng p-value.

## 2.4. Lưu artifact tái lập

```python
!ls -lh outputs/training
```

Phải lưu:

- `mshf_*.joblib`
- `predictions_*.csv`
- `training_summary.json`
- `comparison.csv`
- `statistical_tests.json`
- `run_config.json`
- `source_splits.csv` từ audit

Nén checkpoint định kỳ:

```python
!zip -qr outputs/mshf_stage2_artifacts.zip outputs/audit outputs/training outputs/enhanced_dataset.csv
```

---

# GIAI ĐOẠN 3 — GROUND TRUTH, MANIFEST VÀ LOCALIZATION

## 3.1. Chuẩn hóa attack manifest

```python
!python -m mshf.manifest \
  --source ../Tuan1_2/VEDTD/1.pdfs/manifest.csv \
  --output outputs/manifest/attack_manifest.csv
```

```python
manifest = pd.read_csv('outputs/manifest/attack_manifest.csv')
display(manifest.head())
print(manifest.attack_type.value_counts())
```

Manifest cũ có thể chưa chứa bbox, subtype và severity. Các trường đó phải giữ `unknown`/rỗng cho đến khi có annotation hoặc bộ sinh tấn công cung cấp bằng chứng trực tiếp.

## 3.2. Hoàn thiện subtype và severity

Bốn nhãn chính không đổi. Bổ sung metadata:

```text
insert: character/token/phrase/line/entity/numeric
delete: character/token/phrase/line/non_contiguous
modify: lexical/negation/entity/date/money/obligation
layout: translation/spacing/alignment/font/scale/margin
severity: low/medium/high
```

Sau khi cập nhật generator/manifest, chạy lại audit manifest và lưu seed, page, affected line/token và bbox. Không gán severity bằng suy đoán từ nhãn lớp.

## 3.3. Authentic controls

Tạo JPEG, blur, resize, skew, contrast/noise và perspective từ original, lưu ngoài dữ liệu nguồn:

```text
outputs/controls/<corruption>/<level>/<source_document_id>.*
```

Mỗi control phải có provenance JSON: source, transform, level, seed, output hash. Không ghi đè PDF gốc. Nếu chưa có generator controls chính thức, đánh dấu bước này chưa hoàn thành thay vì dùng file không truy vết.

## 3.4. Localization

```python
!python -m mshf.localize --out outputs/localization
```

```python
display(pd.read_csv('outputs/localization/localization_metrics.csv'))
```

Line F1 được dùng khi có line ground truth. Region IoU/Dice chỉ báo cáo khi manifest có bbox trực tiếp hoặc annotation đã xác minh; không gọi OCR-derived anchor là bbox ground truth.

---

# GIAI ĐOẠN 4 — SEMANTIC CRITICAL CHANGE (B2)

## 4.1. Sinh aligned line pairs trên toàn bộ dữ liệu

```python
!python -m mshf.semantic_features prepare \
  --output outputs/semantic/line_pairs.jsonl
```

```python
pairs = pd.read_json('outputs/semantic/line_pairs.jsonl', lines=True)
print(len(pairs), pairs.source_document_id.nunique())
display(pairs.head())
```

`ref_line/cand_line` đến từ OCR; GT text không được dùng làm input inference.

## 4.2. Chạy Vietnamese NLI/cross-encoder trên GPU

```python
!python -m mshf.colab_nli \
  --pairs outputs/semantic/line_pairs.jsonl \
  --output outputs/semantic/nli_scores.csv \
  --model joeddav/xlm-roberta-large-xnli \
  --batch-size 16
```

Nếu hết VRAM, giảm batch size xuống 4 hoặc 8. Ghi chính xác model revision trong hồ sơ thí nghiệm. Model mặc định là multilingual XNLI baseline; nếu thay bằng Vietnamese NLI model, phải khóa model ID/revision và trích dẫn đúng.

Kiểm tra coverage:

```python
nli = pd.read_csv('outputs/semantic/nli_scores.csv')
assert len(nli) == len(pairs)
assert nli.pair_id.is_unique
print(nli[['contradiction','entailment','neutral']].describe())
```

## 4.3. Aggregate B2 features

```python
!python -m mshf.semantic_features aggregate \
  --pairs outputs/semantic/line_pairs.jsonl \
  --nli outputs/semantic/nli_scores.csv \
  --output outputs/semantic/b2_features.csv
```

B2 gồm contradiction mean/max/p90/high-count, entity, numeric, unit, negation, obligation và logic changes có trọng số OCR confidence.

## 4.4. Join B2 và chạy ablation

```python
base = pd.read_csv('outputs/enhanced_dataset.csv')
b2 = pd.read_csv('outputs/semantic/b2_features.csv')
full = base.merge(b2, on=['source_document_id','category'], how='left')
# Original tự so với chính nó có B2=0; tampered thiếu B2 là lỗi coverage.
b2_cols = [c for c in b2.columns if c.startswith('b2_')]
full.loc[full.category == '1.original', b2_cols] = 0
assert not full.loc[full.category != '1.original', b2_cols].isna().any().any()
full.to_csv('outputs/enhanced_dataset_b2.csv', index=False, encoding='utf-8-sig')
```

Bổ sung B2 vào feature registry/ablation trước khi chạy train. Không gọi A+B1+B2+C là đã hoàn thành nếu chưa chạy full NLI coverage.

Phân tích riêng `modify` và `low severity` chỉ khi manifest đã có severity thật.

---

# GIAI ĐOẠN 5 — THỰC NGHIỆM HOÀN CHỈNH

## 5.1. Robustness matrix

Với từng corruption và level, chạy lại OCR/layout/feature bằng cùng pipeline, sau đó đánh giá bằng checkpoint/split đã khóa. Bảng tối thiểu:

```text
corruption, level, n_sources, macro_f1, balanced_accuracy, auroc, relative_drop
```

Không trộn corrupted variants của một source sang fold khác.

## 5.2. Severity analysis

Join predictions với manifest, group theo `attack_type + severity`, báo n, macro-F1/recall và CI theo source. Nếu severity còn `unknown`, báo `not_available`.

## 5.3. Unseen-subtype/generator

Mỗi lần giữ một subtype hoàn toàn ngoài training. Binary detection là kết quả chính; five-class chỉ đánh giá lớp chính nếu attack type đã được biết. Lưu danh sách train/test source và subtype cho từng run.

## 5.4. Cross-template

Chỉ chạy khi manifest có `template_id` đã xác minh. Toàn bộ source cùng template phải ở cùng phía train/test. Nếu không có metadata, ghi rõ `cross-template: not_available`.

## 5.5. Case studies, calibration và failure taxonomy

Từ OOF predictions và localization details, chọn:

- true positive rõ ràng;
- false positive authentic degradation;
- false negative low severity;
- nhầm insert/delete/modify/layout;
- OCR failure;
- alignment failure;
- semantic/NLI failure;
- geometry failure.

Mỗi case lưu `sample_id`, fold, label, prediction, probabilities, evidence lines, manifest location và lý do phân loại lỗi. Calibration plot phải dùng OOF probabilities.

## 5.6. Đóng gói artifact

```python
!python -m pytest tests -q
!zip -qr outputs/MSHF_REPRODUCIBLE_PACKAGE.zip \
  README.md COLAB_GUIDE.md requirements.txt config.py mshf tests \
  outputs/audit outputs/manifest outputs/training outputs/localization \
  outputs/semantic
```

Tải về:

```python
from google.colab import files
files.download('outputs/MSHF_REPRODUCIBLE_PACKAGE.zip')
```

---

# Checklist trước khi dùng số liệu trong bài báo

- [ ] Audit xác nhận 298 source và 1.490 PDF.
- [ ] Mọi source chỉ thuộc một fold.
- [ ] Split IDs được khóa và dùng chung.
- [ ] Binary không dùng accuracy làm metric chính.
- [ ] OOF stacking không fit meta learner trên in-sample predictions.
- [ ] Threshold/calibration không dùng outer test.
- [ ] Bootstrap theo source.
- [ ] Prediction per sample và config được lưu.
- [ ] Manifest phân biệt annotation trực tiếp và fallback suy từ OCR/GT.
- [ ] NLI có full pair coverage và model revision.
- [ ] Severity/subtype/template không được tự suy diễn.
- [ ] Robustness/unseen/cross-template dùng protocol source-disjoint.
- [ ] Smoke/subset results không xuất hiện như kết quả chính.
- [ ] Quyền dữ liệu và license model/thư viện đã được kiểm tra.
