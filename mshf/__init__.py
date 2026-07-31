"""MSHF (Multi-Scale Hybrid Fusion) — Phát hiện giả mạo tài liệu PDF tiếng Việt.

Cấu trúc package:

  mshf/
  ├── core/       Thư viện nội bộ (feature extraction, model, alignment, I/O)
  ├── cli/        Lệnh chạy hàng ngày (detect, train)
  ├── pipeline/   Chuẩn bị dữ liệu (build_dataset, dataset_audit, manifest…)
  └── research/   Đánh giá nghiên cứu (robustness, cross-template, severity…)

Sử dụng:
  python -m mshf.cli.detect --original goc.pdf --candidate nghi_van.pdf --out-dir outputs/detect/ket_qua
  python -m mshf.cli.train  --dataset outputs/enhanced_dataset.csv --out outputs/training
"""
