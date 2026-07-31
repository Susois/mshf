# HƯỚNG DẪN CHI TIẾT THỰC HIỆN YÊU CẦU 14 VÀ 21-27

## TỔNG QUAN TRẠNG THÁI HIỆN TẠI

| Yêu cầu | Trạng thái | Ghi chú |
|---------|-----------|---------|
| 1-13 | ✅ Hoàn thành | Đã chạy xong |
| **14** | ❌ **CHƯA CÓ LỆNH** | Cần viết script tạo controls |
| 15-20 | ✅ Hoàn thành | B2 features đã tạo |
| **21** | ⚠️ Partial | Có B2 features nhưng chưa integrate vào train.py |
| **22-26** | ❌ **CHƯA TỰ ĐỘNG** | Cần viết runner/script riêng |
| **27** | ✅ Có lệnh | `pytest` + `dataset_audit` sẵn sàng |

---

## YÊU CẦU 14 — TẠO AUTHENTIC CONTROLS (CORRUPTIONS)

### Định nghĩa
Tạo các phiên bản corrupted của PDF gốc (không phải tấn công, mà là các sao chép/transformations tự nhiên) để kiểm tra robustness của mô hình:

- **JPEG recompression** (nén lại): multiple levels
- **Blur** (làm mờ): Gaussian blur, Motion blur
- **Resize/downsampling** (thay đổi kích thước)
- **Skew/rotation** (xoay nhẹ)
- **Contrast/brightness** (điều chỉnh độ sáng)
- **Noise** (thêm nhiễu): Gaussian noise, Salt-pepper
- **Perspective distortion** (biến dạng góc nhìn)

### Cần chuẩn bị
1. Thư viện: Pillow (PIL), OpenCV, pypdf hoặc pdfplumber
2. Dữ liệu input: PDF gốc từ `Tuan1_2/VEDTD/1.pdfs/1.original/`
3. Output directory: `outputs/controls/`

### Code cần viết

Tạo file: `mshf/mshf/create_controls.py`

```python
"""Tạo authentic corrupted PDFs từ original để test robustness."""
import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict, Any
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
import cv2

import config
from . import io_utils


def pdf_to_image(pdf_path: Path) -> np.ndarray:
    """Convert PDF trang đầu thành numpy array."""
    # Dùng PyPDF / pdfplumber / pdf2image
    # Giả sử dùng pdf2image
    from pdf2image import convert_from_path
    images = convert_from_path(pdf_path, first_page=1, last_page=1, dpi=300)
    return np.array(images[0]) if images else None


def image_to_pdf(image: np.ndarray, output_path: Path) -> None:
    """Convert numpy array về PDF."""
    from PIL import Image
    img = Image.fromarray(image)
    img.convert('RGB').save(output_path, 'PDF')


def apply_jpeg_compression(image: np.ndarray, level: int, seed: int = 42) -> np.ndarray:
    """JPEG recompression: level 1-5 (tệ hơn → tốt hơn)"""
    # level=1: quality 40, level=5: quality 95
    quality = 40 + (level - 1) * 15
    img = Image.fromarray(image)
    temp = Path("/tmp/temp_jpeg.jpg")
    img.save(temp, quality=quality)
    result = np.array(Image.open(temp))
    temp.unlink()
    return result


def apply_blur(image: np.ndarray, level: int, seed: int = 42) -> np.ndarray:
    """Gaussian blur: level 1-5 (mờ nhẹ → mờ nặng)"""
    # level=1: kernel=3, level=5: kernel=11
    kernel_size = 3 + (level - 1) * 2
    kernel_size = kernel_size * 2 - 1  # Đảm bảo lẻ
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)


def apply_resize(image: np.ndarray, level: int, seed: int = 42) -> np.ndarray:
    """Downsampling rồi upsample: level 1-5"""
    # level=1: 50% rồi scale lại, level=5: 95% rồi scale lại
    scale = 0.5 + (level - 1) * 0.1
    h, w = image.shape[:2]
    resized = cv2.resize(image, (int(w * scale), int(h * scale)))
    return cv2.resize(resized, (w, h))


def apply_skew(image: np.ndarray, level: int, seed: int = 42) -> np.ndarray:
    """Rotation nhẹ: level 1-5"""
    # level=1: 1 độ, level=5: 5 độ
    angle = level
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REFLECT)


def apply_contrast(image: np.ndarray, level: int, seed: int = 42) -> np.ndarray:
    """Điều chỉnh contrast: level 1-5"""
    # level=1: factor=0.6, level=5: factor=1.4
    factor = 0.6 + (level - 1) * 0.2
    img = Image.fromarray(image)
    enhancer = ImageEnhance.Contrast(img)
    return np.array(enhancer.enhance(factor))


def apply_noise(image: np.ndarray, level: int, seed: int = 42) -> np.ndarray:
    """Thêm Gaussian noise: level 1-5"""
    np.random.seed(seed)
    # level=1: std=5, level=5: std=25
    std = 5 + (level - 1) * 5
    noise = np.random.normal(0, std, image.shape)
    result = np.clip(image.astype(float) + noise, 0, 255).astype(np.uint8)
    return result


def apply_perspective(image: np.ndarray, level: int, seed: int = 42) -> np.ndarray:
    """Perspective distortion: level 1-5"""
    h, w = image.shape[:2]
    # level=1: offset=5px, level=5: offset=25px
    offset = 5 + (level - 1) * 5
    
    pts1 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
    pts2 = np.float32([
        [offset, 0],
        [w - offset, offset],
        [0, h - offset],
        [w - offset, h - offset]
    ])
    
    M = cv2.getPerspectiveTransform(pts1, pts2)
    return cv2.warpPerspective(image, M, (w, h))


def hash_file(path: Path) -> str:
    """SHA256 hash của file."""
    return hashlib.sha256(open(path, 'rb').read()).hexdigest()


def create_corruption(
    source_pdf: Path,
    corruption_type: str,
    level: int,
    output_path: Path,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Tạo corrupted PDF với provenance metadata.
    
    Args:
        source_pdf: đường dẫn PDF gốc
        corruption_type: 'jpeg', 'blur', 'resize', 'skew', 'contrast', 'noise', 'perspective'
        level: 1-5
        output_path: đường dẫn lưu
        seed: random seed
    
    Returns:
        dict với provenance metadata
    """
    # Read source
    image = pdf_to_image(source_pdf)
    if image is None:
        raise ValueError(f"Cannot read {source_pdf}")
    
    # Apply corruption
    corruption_funcs = {
        'jpeg': apply_jpeg_compression,
        'blur': apply_blur,
        'resize': apply_resize,
        'skew': apply_skew,
        'contrast': apply_contrast,
        'noise': apply_noise,
        'perspective': apply_perspective,
    }
    
    if corruption_type not in corruption_funcs:
        raise ValueError(f"Unknown corruption: {corruption_type}")
    
    corrupted = corruption_funcs[corruption_type](image, level, seed)
    
    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image_to_pdf(corrupted, output_path)
    
    # Provenance
    provenance = {
        'source_document_id': source_pdf.stem,
        'corruption': corruption_type,
        'level': level,
        'seed': seed,
        'source_path': str(source_pdf),
        'source_hash': hash_file(source_pdf),
        'output_path': str(output_path),
        'output_hash': hash_file(output_path),
        'parameters': {
            'jpeg': {'quality': 40 + (level - 1) * 15},
            'blur': {'kernel_size': 3 + (level - 1) * 2},
            'resize': {'scale': 0.5 + (level - 1) * 0.1},
            'skew': {'angle': level},
            'contrast': {'factor': 0.6 + (level - 1) * 0.2},
            'noise': {'std': 5 + (level - 1) * 5},
            'perspective': {'offset': 5 + (level - 1) * 5},
        }.get(corruption_type, {})
    }
    
    return provenance


def main():
    ap = argparse.ArgumentParser(description='Tạo authentic corrupted PDFs')
    ap.add_argument('--out-dir', type=Path, default=config.OUTPUT_DIR / 'controls')
    ap.add_argument('--max-docs', type=int, default=0, help='0=tất cả')
    ap.add_argument('--seed', type=int, default=42)
    
    args = ap.parse_args()
    
    # Discover documents
    doc_ids = io_utils.discover_doc_ids()
    if args.max_docs:
        doc_ids = doc_ids[:args.max_docs]
    
    corruptions = ['jpeg', 'blur', 'resize', 'skew', 'contrast', 'noise', 'perspective']
    levels = [1, 2, 3, 4, 5]
    
    manifest = []
    
    for doc_id in doc_ids:
        original_path = config.PDF_ROOT / '1.original' / f'{doc_id}.pdf'
        if not original_path.exists():
            print(f'Missing: {original_path}')
            continue
        
        for corruption in corruptions:
            for level in levels:
                output_path = args.out_dir / corruption / str(level) / f'{doc_id}.pdf'
                try:
                    prov = create_corruption(
                        original_path, corruption, level, output_path, seed=args.seed + level
                    )
                    manifest.append(prov)
                    print(f'✓ {corruption}/{level}/{doc_id}')
                except Exception as e:
                    print(f'✗ {corruption}/{level}/{doc_id}: {e}')
    
    # Save manifest
    import pandas as pd
    df = pd.DataFrame(manifest)
    manifest_path = args.out_dir / 'control_manifest.csv'
    df.to_csv(manifest_path, index=False, encoding='utf-8-sig')
    print(f'\nManifest saved: {manifest_path}')
    print(f'Total corruptions created: {len(manifest)}')


if __name__ == '__main__':
    main()
```

### Cách chạy

```powershell
# Tạo controls cho tất cả documents
python -m mshf.create_controls --out-dir outputs/controls

# Hoặc test với 5 documents đầu
python -m mshf.create_controls --out-dir outputs/controls --max-docs 5
```

### Đầu ra cần có

```
outputs/controls/
├── control_manifest.csv
├── jpeg/
│   ├── 1/
│   ├── 2/
│   └── 5/
├── blur/
├── resize/
├── skew/
├── contrast/
├── noise/
└── perspective/
```

---

## YÊU CẦU 21 — ABLATION B2 VÀ MODIFY/LOW-SEVERITY

### Hiện tại có gì?

✅ `outputs/semantic/b2_features.csv` (từ yêu cầu 20)

### Bước 1: Join B2 vào enhanced_dataset

Tạo file: `mshf/mshf/join_b2.py`

```python
"""Join B2 features vào enhanced dataset."""
import pandas as pd
from pathlib import Path
import config


def main():
    # Load datasets
    base = pd.read_csv(config.OUTPUT_DIR / 'enhanced_dataset.csv')
    b2 = pd.read_csv(config.OUTPUT_DIR / 'semantic' / 'b2_features.csv')
    
    print(f'Base dataset shape: {base.shape}')
    print(f'B2 features shape: {b2.shape}')
    
    # Join theo source_document_id + category
    result = base.merge(
        b2,
        on=['source_document_id', 'category'],
        how='left',
        validate='one_to_one'
    )
    
    print(f'After merge: {result.shape}')
    
    # Fill original documents với 0 (không có semantic changes)
    b2_cols = [c for c in b2.columns if c.startswith('b2_')]
    result.loc[result.category == '1.original', b2_cols] = 0
    
    # Check for missing values
    missing = result[b2_cols].isna().sum().sum()
    if missing > 0:
        raise ValueError(f'Missing B2 features: {missing} cells')
    
    print(f'B2 features joined: {len(b2_cols)} columns')
    
    # Save
    output = config.OUTPUT_DIR / 'enhanced_dataset_b2.csv'
    result.to_csv(output, index=False, encoding='utf-8-sig')
    print(f'Saved to: {output}')


if __name__ == '__main__':
    main()
```

### Bước 2: Update train.py để hỗ trợ B2

Sửa `mshf/mshf/train.py` - tìm dòng 20-21 và thêm:

```python
# THÊM SAU DÒNG 16:
from mshf.semantic_features import B2_FEATURE_COLS

# SỬA DÒNG 20-21 (hàm branches):
def branches(cols):
    return {
        "A": [cols.index(c) for c in config.DOC_FEATURE_COLS if c in cols],
        "B1": [cols.index(c) for c in LINE_FEATURE_COLS if c in cols],
        "C": [cols.index(c) for c in GEOMETRIC_FEATURE_COLS if c in cols],
        "B2": [cols.index(c) for c in B2_FEATURE_COLS if c in cols]  # <-- ADD THIS
    }

# SỬA DÒNG 80 (trong hàm main):
A = config.DOC_FEATURE_COLS
B = LINE_FEATURE_COLS
C = GEOMETRIC_FEATURE_COLS
B2 = B2_FEATURE_COLS  # <-- ADD THIS

# SỬA DÒNG 81 (ablations):
ablations = {
    "A": A,
    "A+B1": A + B,
    "A+C": A + C,
    "B1+C": B + C,
    "A+B1+C": A + B + C,
    "A+B1+B2": A + B + B2,      # <-- ADD THIS
    "A+B1+B2+C": A + B + B2 + C  # <-- ADD THIS
}
```

### Bước 3: Chạy train với B2

```powershell
# Join B2 features trước
python -m mshf.join_b2

# Chạy training với B2
python -m mshf.train `
  --dataset outputs/enhanced_dataset_b2.csv `
  --splits outputs/audit/source_splits.csv `
  --out outputs/training_b2 `
  --task both `
  --models single
```

### Bước 4: Phân tích modify/low-severity (nếu có annotation)

Tạo file: `mshf/mshf/analyze_severity.py`

```python
"""Phân tích kết quả theo severity và attack type."""
import pandas as pd
from sklearn.metrics import f1_score, balanced_accuracy_score, precision_score, recall_score
from pathlib import Path
import config


def main():
    # Load predictions
    pred = pd.read_csv('outputs/training_b2/predictions_is_tampered_A_B1_B2_C_single.csv')
    
    # Load manifest với severity
    manifest = pd.read_csv('outputs/manifest/attack_manifest.csv')
    
    # Check manifest có severity không
    if 'severity' not in manifest.columns:
        print('❌ Manifest không có severity column')
        return
    
    # Join predictions với manifest
    result = pred.merge(
        manifest[['source_document_id', 'attack_type', 'severity']],
        on='source_document_id',
        how='left'
    )
    
    if result['severity'].isna().sum() > 0:
        print(f'⚠️  {result["severity"].isna().sum()} rows missing severity')
    
    # Group analysis
    results = []
    
    for severity in ['low', 'medium', 'high']:
        subset = result[result['severity'] == severity]
        if len(subset) == 0:
            continue
        
        metrics = {
            'severity': severity,
            'count': len(subset),
            'macro_f1': f1_score(subset.y_true, subset.y_pred, average='macro'),
            'balanced_accuracy': balanced_accuracy_score(subset.y_true, subset.y_pred),
            'precision': precision_score(subset.y_true, subset.y_pred, average='macro', zero_division=0),
            'recall': recall_score(subset.y_true, subset.y_pred, average='macro', zero_division=0),
        }
        results.append(metrics)
        print(f"✓ {severity}: n={metrics['count']}, F1={metrics['macro_f1']:.3f}")
    
    # Save
    df = pd.DataFrame(results)
    output = Path('outputs/analysis/severity_metrics_b2.csv')
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False, encoding='utf-8-sig')
    print(f'\nSaved to: {output}')


if __name__ == '__main__':
    main()
```

### Đầu ra cần có

```
outputs/enhanced_dataset_b2.csv
outputs/training_b2/
├── predictions_is_tampered_A_B1_B2_C_single.csv
├── predictions_is_tampered_A_B1_B2_single.csv
├── training_summary.json (với B2 ablations)
└── comparison.csv (với A+B1+B2, A+B1+B2+C)

outputs/analysis/
└── severity_metrics_b2.csv (nếu có annotation)
```

---

## YÊU CẦU 22 — ROBUSTNESS MATRIX

### Định nghĩa
Đánh giá mô hình trên các corrupted controls, đo mức độ suy giảm hiệu suất.

### Cần chuẩn bị
- Controls từ yêu cầu 14
- Trained model từ yêu cầu 1-11: `outputs/training/mshf_is_tampered.joblib`

### Code cần viết

Tạo file: `mshf/mshf/robustness_eval.py`

```python
"""Đánh giá robustness trên corrupted controls."""
import argparse
from pathlib import Path
from typing import Dict, Any
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score, balanced_accuracy_score

import config
from . import io_utils


def extract_features_for_corruption(corruption_type: str, level: int):
    """
    Extract features cho một corruption type + level.
    Cần tạo OCR, layout, features cho controls.
    """
    # TODO: Implement based on build_dataset.py logic
    # Tham khảo: mshf/build_dataset.py
    pass


def evaluate_robustness(checkpoint_path: Path, corruption_type: str, level: int, baseline_f1: float):
    """
    Đánh giá performance trên corruption.
    
    Args:
        checkpoint_path: đường dẫn .joblib model
        corruption_type: 'jpeg', 'blur', etc.
        level: 1-5
        baseline_f1: F1 score trên clean data
    
    Returns:
        dict với metrics
    """
    # Load model
    checkpoint = joblib.load(checkpoint_path)
    model = checkpoint['model']
    features = checkpoint['features']
    
    # Load corrupted features
    try:
        features_df = extract_features_for_corruption(corruption_type, level)
    except Exception as e:
        print(f'Error extracting features for {corruption_type}/{level}: {e}')
        return None
    
    # Prepare X, y
    X = features_df[features].to_numpy(float)
    y = features_df['is_tampered'].to_numpy(int)
    
    # Predict
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]
    
    # Metrics
    f1 = f1_score(y, y_pred, average='macro')
    auroc = roc_auc_score(y, y_proba)
    balanced_acc = balanced_accuracy_score(y, y_pred)
    
    result = {
        'corruption': corruption_type,
        'level': level,
        'sample_count': len(y),
        'macro_f1': f1,
        'balanced_accuracy': balanced_acc,
        'auroc': auroc,
        'relative_drop_f1': (baseline_f1 - f1) / baseline_f1 * 100 if baseline_f1 > 0 else 0,
    }
    
    return result


def main():
    ap = argparse.ArgumentParser(description='Evaluate robustness on corrupted PDFs')
    ap.add_argument(
        '--checkpoint',
        type=Path,
        default=config.OUTPUT_DIR / 'training' / 'mshf_is_tampered.joblib'
    )
    ap.add_argument('--controls-dir', type=Path, default=config.OUTPUT_DIR / 'controls')
    ap.add_argument('--out-dir', type=Path, default=config.OUTPUT_DIR / 'robustness')
    ap.add_argument('--baseline-f1', type=float, default=None, help='Baseline F1 score')
    
    args = ap.parse_args()
    
    # Load baseline F1
    if args.baseline_f1 is None:
        summary_path = config.OUTPUT_DIR / 'training' / 'training_summary.json'
        if summary_path.exists():
            import json
            summary = json.load(open(summary_path))
            for run in summary:
                if run['target'] == 'is_tampered' and run['ablation'] == 'A+B1+C':
                    args.baseline_f1 = run['macro_f1']
                    break
    
    if args.baseline_f1 is None:
        args.baseline_f1 = 0.8
        print(f'⚠️  Using default baseline F1: {args.baseline_f1}')
    
    # Evaluate each corruption
    corruptions = ['jpeg', 'blur', 'resize', 'skew', 'contrast', 'noise', 'perspective']
    levels = [1, 2, 3, 4, 5]
    
    results = []
    
    for corruption in corruptions:
        for level in levels:
            try:
                result = evaluate_robustness(
                    args.checkpoint,
                    corruption,
                    level,
                    args.baseline_f1
                )
                if result:
                    results.append(result)
                    print(f'✓ {corruption}/{level}')
            except Exception as e:
                print(f'✗ {corruption}/{level}: {e}')
    
    # Save results
    args.out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(results)
    output_path = args.out_dir / 'robustness_matrix.csv'
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f'\nRobustness matrix saved: {output_path}')


if __name__ == '__main__':
    main()
```

### Cách chạy

```powershell
python -m mshf.robustness_eval `
  --checkpoint outputs/training/mshf_is_tampered.joblib `
  --controls-dir outputs/controls `
  --out-dir outputs/robustness
```

### Đầu ra

```
outputs/robustness/robustness_matrix.csv
```

Schema:
```
corruption,level,sample_count,macro_f1,balanced_accuracy,auroc,relative_drop_f1
```

---

## YÊU CẦU 23 — SEVERITY ANALYSIS

### Điều kiện
✅ Manifest **phải có** `severity` field với giá trị thật (low/medium/high)

### Code

```python
# mshf/mshf/analyze_severity.py (đã viết ở yêu cầu 21)
```

### Cách chạy

```powershell
python -m mshf.analyze_severity
```

### Đầu ra

```
outputs/analysis/severity_metrics_b2.csv
```

---

## YÊU CẦU 24 — UNSEEN-SUBTYPE/GENERATOR

### Điều kiện
✅ Manifest **phải có** `attack_subtype` và `generator_id`

### Ý tưởng
- Leave-one-generator-out: mỗi lần train trên tất cả generators trừ 1, test trên generator đó
- Binary detection là kết quả chính

### Code outline

```python
# mshf/mshf/unseen_eval.py
from sklearn.model_selection import LeaveOneGroupOut

for generator_id in manifest['generator_id'].unique():
    train_mask = manifest['generator_id'] != generator_id
    test_mask = manifest['generator_id'] == generator_id
    
    # Train model on train_mask
    # Predict on test_mask
    # Save results
```

### Đầu ra

```
outputs/unseen/unseen_generator_metrics.csv
outputs/unseen/predictions_holdout_generator_*.csv
```

---

## YÊU CẦU 25 — CROSS-TEMPLATE EVALUATION

### Điều kiện
✅ Manifest **phải có** `template_id`

### Ý tưởng
Dùng `LeaveOneGroupOut(group=template_id)` để train/test theo template

---

## YÊU CẦU 26 — CASE STUDIES & CALIBRATION PLOT

### Cần chuẩn bị
- `outputs/training/predictions_*_stacking.csv` (OOF predictions)
- `outputs/localization/localization_details.json`

### Code cần viết

```python
# mshf/mshf/create_report.py
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
import pandas as pd

# Load OOF predictions
pred = pd.read_csv('outputs/training/predictions_is_tampered_A_B1_C_stacking.csv')

# Calibration plot
prob_true, prob_pred = calibration_curve(
    pred.y_true,
    pred.proba_tampered,
    n_bins=10,
    strategy='uniform'
)

plt.figure(figsize=(8, 6))
plt.plot(prob_pred, prob_true, 'o-', label='Mô hình')
plt.plot([0, 1], [0, 1], 'k--', label='Hoàn hảo')
plt.xlabel('Xác suất dự đoán trung bình')
plt.ylabel('Tỷ lệ positives thực')
plt.legend()
plt.savefig('outputs/report/calibration_plot.png', dpi=150)

# Case studies