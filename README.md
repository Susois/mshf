# MSHF — Multi-Scale Hybrid Fusion

Hệ thống phát hiện và định vị giả mạo tài liệu PDF tiếng Việt.

Cho trước một **bản gốc** và một **bản nghi vấn**, MSHF phân loại một trong 5 nhãn:

| Nhãn | Mô tả |
|---|---|
| `insert` | Chèn thêm đoạn văn |
| `delete` | Xóa một số dòng |
| `modify` | Sửa nội dung (đổi từ, số liệu…) |
| `layout` | Thay đổi định dạng (font, cỡ chữ, giãn dòng) |
| `original` | Không thay đổi |

Ngoài phân loại, hệ thống còn **định vị từng dòng bị can thiệp** và xuất báo cáo HTML trực quan.

**Kết quả đã đạt:**

| Task | Cấu hình | Macro F1 |
|---|---|---|
| Binary (tampered / authentic) | A+B1+C | **98.86%** |
| 5-class | A+B1+C / stacking | **98.52%** |

---

## Cài đặt

### 1. Clone repo

```bash
git clone https://github.com/Susois/mshf.git
cd mshf
```

### 2. Tạo môi trường ảo và cài thư viện

```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

Yêu cầu **Python 3.10+**.

### 3. Tải model đã train

Xem mục **[Model đã train](#model-đã-train)** bên dưới để tải `mshf_label.joblib` và đặt vào `outputs/training/`.

---

## Sử dụng

### Chạy phát hiện giả mạo

```powershell
cd C:\Su\Dean3\mshf

python -m mshf.detect `
  --original  "C:\đường\dẫn\tới\van_ban_goc.pdf" `
  --candidate "C:\đường\dẫn\tới\van_ban_nghi_van.pdf" `
  --out-dir   "outputs\detect\ten_tai_lieu" `
  --mode full
```

**`--mode full` là chế độ đầy đủ và chính xác nhất** — dùng toàn bộ 3 nhánh feature A+B1+C (PhoBERT + LayoutLMv3 + geometric). Đây là chế độ khuyến nghị.

Lần đầu chạy, PhoBERT (~400 MB) và LayoutLMv3 sẽ được tải tự động về `~/.cache/huggingface/`. Từ lần thứ hai trở đi sẽ dùng cache, không cần tải lại.

---

### Ví dụ cụ thể

```powershell
cd C:\Su\Dean3\mshf

python -m mshf.detect `
  --original  "C:\Su\DeAn\Tuan6\detector_and_explainer\1.pdfs\1.original\HoangVanNam_11236160.pdf" `
  --candidate "C:\Su\DeAn\Tuan6\detector_and_explainer\1.pdfs\2.insert\HoangVanNam_11236160.pdf" `
  --out-dir   "outputs\detect\HoangVanNam_insert" `
  --mode full
```

---

### Kết quả output

Sau khi chạy xong, thư mục `--out-dir` chứa:

```
outputs/detect/HoangVanNam_insert/
├── tampered_report.html      ← mở bằng trình duyệt để xem kết quả trực quan
├── report.json               ← kết quả đầy đủ dạng JSON
├── page_000_highlighted.png  ← trang 1 với các dòng bị can thiệp được tô màu
└── page_001_highlighted.png  ← ...
```

Mở báo cáo:

```powershell
start outputs\detect\HoangVanNam_insert\tampered_report.html
```

**Màu sắc highlight:**

| Màu | Loại | Ý nghĩa |
|---|---|---|
| 🔴 Đỏ | `inserted` | Dòng chèn thêm, không có trong bản gốc |
| 🟠 Cam | `deleted` | Dòng bị xóa, còn trong bản gốc |
| 🟡 Vàng | `modified` | Dòng bị sửa nội dung |

---

### Nếu máy không chạy được `--mode full`

Nếu máy thiếu RAM hoặc không tải được model PhoBERT/LayoutLMv3, thêm `--mode fast` để bỏ qua nhánh A:

```powershell
python -m mshf.detect `
  --original  "goc.pdf" `
  --candidate "nghi_van.pdf" `
  --out-dir   "outputs\detect\ket_qua" `
  --mode fast
```

Chế độ `fast` chỉ dùng B1+C features (~5 giây/cặp, không cần internet), vẫn chính xác với `insert` / `delete` / `modify`. Độ chính xác với `layout` thấp hơn.

---

### PDF scan (không có text nhúng)

Nếu PDF là bản scan (chụp ảnh), thêm `--ocr paddleocr`:

```powershell
python -m mshf.detect `
  --original  "goc.pdf" `
  --candidate "nghi_van.pdf" `
  --out-dir   "outputs\detect\ket_qua" `
  --mode full `
  --ocr paddleocr
```

---

## Đọc kết quả JSON

```python
import json

with open("outputs/detect/HoangVanNam_insert/report.json", encoding="utf-8") as f:
    report = json.load(f)

print(report["verdict"])                        # "TAMPERED" hoặc "AUTHENTIC"
print(report["model_prediction"]["label"])      # "insert"
print(report["model_prediction"]["confidence"]) # 0.9994
print(report["total_tampered_lines"])           # số dòng bị can thiệp

for page in report["pages"]:
    if page["tampered_lines"]:
        print(f"Trang {page['page_index'] + 1}:")
        for line in page["tampered_lines"]:
            print(f"  [{line['type'].upper()}] {line['cand_text'][:80]}")
```

---

## Model đã train

Tải model tại: **[Releases → mshf_label.joblib](https://github.com/Susois/mshf/releases/latest)**

```powershell
mkdir outputs\training
Invoke-WebRequest -Uri "https://github.com/Susois/mshf/releases/latest/download/mshf_label.joblib" `
  -OutFile "outputs\training\mshf_label.joblib"
```

Đọc model trong Python:

```python
import joblib
artifact = joblib.load("outputs/training/mshf_label.joblib")
# artifact["model"]    → XGBoost model
# artifact["labels"]   → ["delete", "insert", "layout", "modify", "original"]
# artifact["features"] → danh sách 40 feature columns
```

```
mshf/
├── mshf/
│   ├── core/        thư viện nội bộ (feature extraction, alignment, model, I/O)
│   ├── cli/         lệnh chạy (detect, train)
│   ├── pipeline/    chuẩn bị dữ liệu (build_dataset, dataset_audit…)
│   └── research/    đánh giá nghiên cứu (robustness, cross-template…)
├── outputs/
│   └── training/    model đã train (mshf_label.joblib)
├── tests/
├── config.py
└── requirements.txt
```

---

## Yêu cầu hệ thống

- Python 3.10+
- RAM: tối thiểu 8 GB (khuyến nghị 16 GB cho `--mode full`)
- Dung lượng: ~1 GB cho cache HuggingFace (PhoBERT + LayoutLMv3)
- GPU (tùy chọn): tăng tốc nhánh A đáng kể, nhưng không bắt buộc
