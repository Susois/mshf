# sample_docs — Thư mục mẫu để chạy thử

Đặt PDF của bạn vào đây theo cấu trúc:

```
sample_docs/
├── 1.original/   ← bản gốc tham chiếu
├── 2.insert/     ← bản nghi vấn bị chèn thêm nội dung
├── 3.delete/     ← bản nghi vấn bị xóa nội dung
├── 4.modify/     ← bản nghi vấn bị sửa nội dung
└── 5.layout/     ← bản nghi vấn bị thay đổi định dạng
```

**Quy tắc:** file gốc và file nghi vấn phải **cùng tên** nhau.

Ví dụ:
```
sample_docs/
├── 1.original/my_document.pdf
└── 2.insert/my_document.pdf    ← cùng tên với bản gốc
```

## Chạy phát hiện

```powershell
cd C:\path\to\mshf

python -m mshf.detect `
  --original  "sample_docs\1.original\my_document.pdf" `
  --candidate "sample_docs\2.insert\my_document.pdf" `
  --out-dir   "outputs\detect\my_document_insert" `
  --mode full
```

Kết quả xuất vào `outputs\detect\my_document_insert\`:
- `tampered_report.html` — mở bằng trình duyệt
- `report.json` — kết quả JSON
- `page_000_highlighted.png` — ảnh trang với highlight màu
