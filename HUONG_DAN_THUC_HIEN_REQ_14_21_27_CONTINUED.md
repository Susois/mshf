# PHẦN TIẾP THEO: YÊU CẦU 26 VÀ 27 (TIẾP THEO)

## YÊU CẦU 26 — CASE STUDIES & CALIBRATION PLOT (TIẾP THEO)

### Code đầy đủ cho mshf/mshf/create_report.py

```python
"""Tạo calibration plot, case studies và failure taxonomy."""
import argparse
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict, Any
import json
import pandas as pd
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import confusion_matrix, classification_report

import config


def create_calibration_plot(predictions_df: pd.DataFrame, output_path: Path):
    """Tạo calibration plot từ OOF predictions."""
    prob_true, prob_pred = calibration_curve(
        predictions_df.y_true,
        predictions_df.proba_tampered,
        n_bins=10,
        strategy='uniform'
    )
    
    plt.figure(figsize=(8, 6))
    plt.plot(prob_pred, prob_true, 'o-', linewidth=2, label='Mô hình Stacking')
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Hoàn hảo')
    plt.xlabel('Xác suất dự đoán trung bình')
    plt.ylabel('Tỷ lệ positives thực')
    plt.title('Biểu đồ Calibration - Phát hiện giả mạo')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    print(f'✓ Calibration plot saved: {output_path}')
    
    # Save calibration metrics
    bins_df = pd.DataFrame({
        'mean_predicted_prob': prob_pred,
        'fraction_of_positives': prob_true,
    })
    bins_df.to_csv(output_path.parent / 'calibration_bins.csv', index=False)


def create_case_studies(predictions_df: pd.DataFrame, output_path: Path, n_per_group: int = 5):
    """Chọn TP, FP, FN đại diện."""
    cases = []
    
    # True Positives
    tp = predictions_df[(predictions_df.y_true == 1) & (predictions_df.y_pred == 1)]
    tp_sorted = tp.sort_values('proba_tampered', ascending=False)
    for idx, row in tp_sorted.head(n_per_group).iterrows():
        cases.append({
            'case_type': 'TP',
            'sample_id': row['sample_id'],
            'source_document_id': row['source_document_id'],
            'y_true': row['y_true'],
            'y_pred': row['y_pred'],
            'confidence': row['proba_tampered'],
            'reason': 'Phát hiện đúng tấn công'
        })
    
    # False Positives
    fp = predictions_df[(predictions_df.y_true == 0) & (predictions_df.y_pred == 1)]
    fp_sorted = fp.sort_values('proba_tampered', ascending=False)
    for idx, row in fp_sorted.head(n_per_group).iterrows():
        cases.append({
            'case_type': 'FP',
            'sample_id': row['sample_id'],
            'source_document_id': row['source_document_id'],
            'y_true': row['y_true'],
            'y_pred': row['y_pred'],
            'confidence': row['proba_tampered'],
            'reason': 'Cảnh báo sai - tài liệu sạch'
        })
    
    # False Negatives
    fn = predictions_df[(predictions_df.y_true == 1) & (predictions_df.y_pred == 0)]
    fn_sorted = fn.sort_values('proba_tampered', ascending=True)
    for idx, row in fn_sorted.head(n_per_group).iterrows():
        cases.append({
            'case_type': 'FN',
            'sample_id': row['sample_id'],
            'source_document_id': row['source_document_id'],
            'y_true': row['y_true'],
            'y_pred': row['y_pred'],
            'confidence': row['proba_tampered'],
            'reason': 'Bỏ sót tấn công'
        })
    
    df = pd.DataFrame(cases)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f'✓ Case studies saved: {output_path} ({len(cases)} cases)')


def create_failure_taxonomy(predictions_df: pd.DataFrame, localization_path: Path, output_path: Path):
    """Tạo taxonomy của các lỗi dự đoán."""
    # Load localization details
    if localization_path.exists():
        with open(localization_path) as f:
            loc_details = json.load(f)
    else:
        loc_details = {}
    
    # Collect failures
    failures = []
    
    # Errors
    errors = predictions_df[predictions_df.y_true != predictions_df.y_pred]
    
    for idx, row in errors.iterrows():
        error_type = 'False Positive' if row['y_true'] == 0 else 'False Negative'
        
        # Try to find localization details
        doc_id = str(row['source_document_id'])
        loc_info = loc_details.get(doc_id, {})
        
        failure = {
            'sample_id': row['sample_id'],
            'source_document_id': row['source_document_id'],
            'error_type': error_type,
            'y_true': row['y_true'],
            'y_pred': row['y_pred'],
            'confidence': row['proba_tampered'],
            'possible_cause': classify_failure_cause(row, loc_info),
            'localization_available': bool(loc_info)
        }
        failures.append(failure)
    
    df = pd.DataFrame(failures)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f'✓ Failure taxonomy saved: {output_path} ({len(failures)} errors)')


def classify_failure_cause(row: pd.Series, loc_info: Dict) -> str:
    """Phân loại nguyên nhân lỗi."""
    # Heuristic classification
    if row['y_true'] == 0 and row['y_pred'] == 1:
        return 'OCR degradation trên authentic'
    elif row['y_true'] == 1 and row['y_pred'] == 0:
        confidence_rank = row['proba_tampered']
        if confidence_rank < 0.3:
            return 'Low semantic difference'
        else:
            return 'Feature extraction failure'
    return 'Unknown'


def main():
    ap = argparse.ArgumentParser(description='Tạo báo cáo và visualizations')
    ap.add_argument(
        '--predictions',
        type=Path,
        default=config.OUTPUT_DIR / 'training' / 'predictions_is_tampered_A_B1_C_stacking.csv'
    )
    ap.add_argument(
        '--localization',
        type=Path,
        default=config.OUTPUT_DIR / 'localization' / 'localization_details.json'
    )
    ap.add_argument('--out-dir', type=Path, default=config.OUTPUT_DIR / 'report')
    
    args = ap.parse_args()
    
    # Load predictions
    pred_df = pd.read_csv(args.predictions)
    print(f'Loaded {len(pred_df)} predictions')
    
    # Create visualizations
    args.out_dir.mkdir(parents=True, exist_ok=True)
    
    create_calibration_plot(
        pred_df,
        args.out_dir / 'calibration_plot.png'
    )
    
    create_case_studies(
        pred_df,
        args.out_dir / 'case_studies.csv'
    )
    
    create_failure_taxonomy(
        pred_df,
        args.localization,
        args.out_dir / 'failure_taxonomy.csv'
    )
    
    print(f'\n✓ All reports saved to: {args.out_dir}')


if __name__ == '__main__':
    main()
```

### Cách chạy

```powershell
python -m mshf.create_report `
  --predictions outputs/training/predictions_is_tampered_A_B1_C_stacking.csv `
  --localization outputs/localization/localization_details.json `
  --out-dir outputs/report
```

### Đầu ra

```
outputs/report/
├── calibration_plot.png
├── calibration_bins.csv
├── case_studies.csv (TP, FP, FN examples)
└── failure_taxonomy.csv (phân loại lỗi)
```

---

## YÊU CẦU 27 — CHUẨN HÓA MÃ NGUỒN VÀ TÁI LẬP

### Lệnh chạy từ mshf root

```powershell
# Chạy tests
python -m pytest tests -q

# Chạy audit strict
python -m mshf.dataset_audit --strict --folds 5

# Lưu log
mkdir outputs/requirement_27
python -m pytest tests -q > outputs/requirement_27/tests.txt 2>&1
python -m mshf.dataset_audit --strict --folds 5 > outputs/requirement_27/audit.txt 2>&1
```

### Lệnh Colab

```python
%cd /content/mshf_repo/mshf
!python -m pytest tests -q > outputs/requirement_27/tests.txt 2>&1
!python -m mshf.dataset_audit --strict --folds 5 > outputs/requirement_27/audit.txt 2>&1
```

### Cần lưu kèm

```
mshf/
├── requirements.txt
├── .env.example
├── config.py
├── mshf/
├── tests/

outputs/
├── audit/
│   └── source_splits.csv (khóa lần đầu)
├── enhanced_dataset.csv
├── training/
│   ├── run_config.json
│   ├── predictions_*.csv
│   ├── mshf_*.joblib
│   └── training_summary.json
├── training_b2/
│   └── predictions_*.csv (nếu làm yêu cầu 21)
└── requirement_27/
    ├── tests.txt
    └── audit.txt
```

### Điều kiện hoàn thành

- ✅ `python -m pytest tests -q` pass
- ✅ `dataset_audit --strict` xác nhận 298 source / 1.490 samples
- ✅ Lưu tất cả predictions per sample
- ✅ Lưu run_config.json với seed, versions, paths
- ✅ Có thể rerun từ source + data + instructions

---

## THỨ TỰ THỰC HIỆN (BẮTBUỘC CÓ THỨ TỰ)

```
1️⃣ Yêu cầu 14
   ├─ Viết mshf/mshf/create_controls.py
   ├─ Chạy: python -m mshf.create_controls
   └─ Output: outputs/controls/ + control_manifest.csv

2️⃣ Yêu cầu 21
   ├─ Viết mshf/mshf/join_b2.py
   ├─ Sửa mshf/mshf/train.py thêm B2
   ├─ Chạy: python -m mshf.join_b2
   ├─ Chạy: python -m mshf.train --dataset enhanced_dataset_b2.csv
   └─ Output: outputs/training_b2/

3️⃣ Yêu cầu 22
   ├─ Viết mshf/mshf/robustness_eval.py
   ├─ Chạy: python -m mshf.robustness_eval
   └─ Output: outputs/robustness/robustness_matrix.csv

4️⃣ Yêu cầu 23 (điều kiện: có severity)
   ├─ Viết mshf/mshf/analyze_severity.py
   ├─ Chạy: python -m mshf.analyze_severity
   └─ Output: outputs/analysis/severity_metrics_b2.csv

5️⃣ Yêu cầu 24-25 (điều kiện: có metadata)
   ├─ Viết unseen_eval.py / cross_template_eval.py
   └─ Output: outputs/unseen/ hoặc outputs/cross_template/

6️⃣ Yêu cầu 26
   ├─ Viết mshf/mshf/create_report.py
   ├─ Chạy: python -m mshf.create_report
   └─ Output: outputs/report/ (plot + tables)

7️⃣ Yêu cầu 27
   ├─ Chạy: pytest + audit
   ├─ Lưu logs
   └─ Đóng gói artifacts
```

---

## CHECKLIST CHUẨN BỊ

### Dữ liệu cần có (từ yêu cầu 1-20)
- [x] outputs/enhanced_dataset.csv (1.490 rows × 45 cols)
- [x] outputs/semantic/b2_features.csv (1.490 rows × 10 B2 cols)
- [x] outputs/training/predictions_*.csv (OOF predictions)
- [x] outputs/audit/source_splits.csv (fixed splits khóa)
- [x] outputs/training/mshf_is_tampered.joblib (trained model)

### Cần viết thêm (Yêu cầu 14-26)
- [ ] mshf/mshf/create_controls.py (Req 14)
- [ ] mshf/mshf/join_b2.py (Req 21)
- [ ] Sửa mshf/mshf/train.py (Req 21)
- [ ] mshf/mshf/analyze_severity.py (Req 21/23)
- [ ] mshf/mshf/robustness_eval.py (Req 22)
- [ ] mshf/mshf/unseen_eval.py (Req 24)
- [ ] mshf/mshf/cross_template_eval.py (Req 25)
- [ ] mshf/mshf/create_report.py (Req 26)

### Metadata cần annotation (nếu muốn làm 23-25)
- [ ] Manifest có `severity` (low/medium/high)
- [ ] Manifest có `attack_subtype`
- [ ] Manifest có `generator_id`
- [ ] Manifest có `template_id`

### Lưu trữ (Req 27)
- [ ] requirements.txt
- [ ] .env.example
- [ ] outputs/audit/source_splits.csv (khóa)
- [ ] outputs/training/run_config.json
- [ ] Tất cả predictions_*.csv
- [ ] requirements_27/tests.txt
- [ ] requirements_27/audit.txt

---

## QUICK START

### Cách chạy nhanh (test 5 docs)

```powershell
# 1. Tạo controls (5 docs)
python -m mshf.create_controls --max-docs 5

# 2. Join B2
python -m mshf.join_b2

# 3. Train với B2
python -m mshf.train --dataset outputs/enhanced_dataset_b2.csv --out outputs/training_b2_test

# 4. Tạo report
python -m mshf.create_report --out-dir outputs/report_test
```

### Cách chạy đầy đủ (tất cả 298 docs)

```powershell
# 1. Tạo controls (tất cả)
python -m mshf.create_controls

# 2. Extract OCR/features cho controls
# (Cần implement trong robustness_eval.py)

# 3. Robustness evaluation
python -m mshf.robustness_eval

# 4. Final validation
python -m pytest tests -q
python -m mshf.dataset_audit --strict --folds 5
```

---

## CÁC LỖI THƯỜNG GẶP

### Error: "Missing B2 features"
```
→ Kiểm tra: outputs/semantic/b2_features.csv có tồn tại không
→ Kiểm tra: số dòng và columns có match không
→ Re-run: python -m mshf.semantic_features aggregate
```

### Error: "Split file does not cover every source"
```
→ Kiểm tra: outputs/audit/source_splits.csv bị mất
→ Re-run: python -m mshf.dataset_audit --strict --folds 5
```

### Error: "Missing corrupted PDFs"
```
→ Kiểm tra: outputs/controls/ directory
→ Re-run: python -m mshf.create_controls
```

### Error: PDF to image conversion fails
```
→ Cài đặt: pip install pdf2image pillow opencv-python
→ Kiểm tra: pdfs có valid không
```

---

## GHI CHÚ QUAN TRỌNG

1. **Không đổi split**: Dùng `source_splits.csv` từ lần đầu cho tất cả experiments
2. **Không điền fake data**: Nếu severity/subtype chưa có, báo `unknown`
3. **Lưu seeds**: Tất cả experiments phải lưu seed để reproduce
4. **Lưu configs**: Mỗi run phải lưu run_config.json
5. **Kiểm tra coverage**: Tất cả 298 sources phải có kết quả
6. **Dùng OOF predictions**: Không dùng training predictions cho metrics

---

**Chúc bạn thực hiện thành công!**