<div align="center">

# MSHF — Multi-Scale Hybrid Fusion

**Phát hiện và định vị giả mạo tài liệu PDF tiếng Việt**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![GitHub Release](https://img.shields.io/github/v/release/Susois/mshf)](https://github.com/Susois/mshf/releases/latest)

[English](README_EN.md) · [Hướng dẫn chi tiết](HUONG_DAN_SU_DUNG.md)

</div>

---

## Giới thiệu

MSHF là hệ thống phát hiện giả mạo tài liệu PDF tiếng Việt sử dụng kiến trúc **Multi-Scale Hybrid Fusion** — kết hợp 3 nhánh đặc trưng độc lập:

- **Nhánh A** — Semantic: CER/WER + PhoBERT + LayoutLMv3 *(8 features)*
- **Nhánh B1** — Line-level: đếm insert/delete/modify theo từng dòng *(16 features)*
- **Nhánh C** — Geometric: delta font/spacing/margin + residual *(16 features)*

Tất cả hội tụ vào **XGBoost** phân loại 5 nhãn: `original` · `insert` · `delete` · `modify` · `layout`

Ngoài phân loại, hệ thống còn **định vị từng dòng bị can thiệp** trên từng trang và xuất báo cáo HTML trực quan.

---

## Kết quả

| Task | Cấu hình | Macro F1 | Balanced Acc |
|---|---|---|---|
| Binary (tampered/authentic) | A+B1+C / single | **98.86%** | 99.41% |
| 5-class | A+B1+C / stacking | **98.52%** | 98.52% |

Đánh giá trên bộ dữ liệu **VEDTD** (1.490 tài liệu pháp luật, 5-fold source-disjoint cross-validation).

---

## Cài đặt

**Yêu cầu:** Python 3.10+ · RAM ≥ 8 GB · ~1 GB dung lượng cho model cache

```bash
git clone https://github.com/Susois/mshf.git
cd mshf
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

### Tải model

Model được phân phối qua **[GitHub Releases](https://github.com/Susois/mshf/releases/latest)**.

```powershell
mkdir outputs\training
Invoke-WebRequest `
  -Uri "https://github.com/Susois/mshf/releases/latest/download/mshf_label.joblib" `
  -OutFile "outputs\training\mshf_label.joblib"
```

---

## Sử dụng nhanh

Đặt PDF vào `sample_docs/` theo cấu trúc:

```
sample_docs/
├── 1.original/  ← bản gốc
├── 2.insert/    ← bản nghi vấn (cùng tên file với bản gốc)
├── 3.delete/
├── 4.modify/
└── 5.layout/
```

Chạy phát hiện:

```powershell
python -m mshf.detect `
  --original  "sample_docs\1.original\ten_tai_lieu.pdf" `
  --candidate "sample_docs\2.insert\ten_tai_lieu.pdf" `
  --out-dir   "outputs\detect\ket_qua" `
  --mode full
```

> **`--mode full`** dùng toàn bộ A+B1+C — chính xác nhất.  
> Lần đầu chạy sẽ tải PhoBERT (~400 MB) và LayoutLMv3 tự động.  
> Nếu máy yếu RAM, dùng `--mode fast` (B1+C, ~5 giây, không cần internet).

### Kết quả

```
outputs/detect/ket_qua/
├── tampered_report.html   ← mở bằng trình duyệt
├── report.json            ← kết quả JSON đầy đủ
└── page_000_highlighted.png
```

| Màu | Ý nghĩa |
|---|---|
| 🔴 Đỏ | Dòng chèn thêm (`inserted`) |
| 🟠 Cam | Dòng bị xóa (`deleted`) |
| 🟡 Vàng | Dòng bị sửa (`modified`) |

### Tùy chọn

| Tình huống | Tham số |
|---|---|
| PDF scan (không có text) | `--ocr paddleocr` |
| Máy không đủ RAM | `--mode fast` |
| Chỉ định model khác | `--model path/to/mshf_label.joblib` |

---

## Đọc kết quả bằng Python

```python
from pathlib import Path
from mshf.detect import detect

report = detect(
    original_pdf=Path("sample_docs/1.original/doc.pdf"),
    candidate_pdf=Path("sample_docs/2.insert/doc.pdf"),
    out_dir=Path("outputs/detect/doc_insert"),
)

print(report["verdict"])                         # "TAMPERED" | "AUTHENTIC"
print(report["model_prediction"]["label"])       # "insert"
print(report["model_prediction"]["confidence"])  # 0.9994
print(report["total_tampered_lines"])            # 9

for page in report["pages"]:
    for line in page["tampered_lines"]:
        print(f"[{line['type'].upper()}] {line['cand_text'][:80]}")
```

---

## Cấu trúc repo

```
mshf/
├── mshf/
│   ├── core/          thư viện nội bộ (features, alignment, model, I/O)
│   ├── cli/           lệnh chạy (detect, train)
│   ├── pipeline/      chuẩn bị dữ liệu
│   └── research/      đánh giá nghiên cứu
├── scripts/
│   ├── baseline/      baseline Tuần 6 (RF, PhoBERT, LayoutLMv3...)
│   ├── build_dataset/ sinh dữ liệu VEDTD
│   └── evaluation/    đánh giá OCR, semantic
├── data/
│   └── hybrid_fusion_dataset.csv   dataset nhánh A
├── results/           kết quả so sánh baseline
├── sample_docs/       đặt PDF vào đây để chạy thử
├── outputs/           kết quả detect, model
├── tests/
├── config.py
└── requirements.txt
```

---

## License

Dự án này được phân phối theo giấy phép [MIT](LICENSE).  
Xem [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md) để biết thông tin về các thư viện bên thứ ba.
