<div align="center">

# MSHF — Multi-Scale Hybrid Fusion

**Vietnamese Legal Document Tampering Detection and Localization**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![GitHub Release](https://img.shields.io/github/v/release/Susois/mshf)](https://github.com/Susois/mshf/releases/latest)

[Tiếng Việt](README.md)

</div>

---

## Overview

MSHF is a PDF document tampering detection system for Vietnamese legal texts, built on a **Multi-Scale Hybrid Fusion** architecture that combines three independent feature branches:

- **Branch A** — Semantic: CER/WER + PhoBERT + LayoutLMv3 *(8 features)*
- **Branch B1** — Line-level: insert/delete/modify counts and ratios *(16 features)*
- **Branch C** — Geometric: delta font/spacing/margin + positional residual *(16 features)*

All branches feed into an **XGBoost** classifier with 5 output labels:  
`original` · `insert` · `delete` · `modify` · `layout`

Beyond classification, the system **localizes each tampered line** per page and generates an interactive HTML report with color-coded highlights.

---

## Results

| Task | Configuration | Macro F1 | Balanced Acc |
|---|---|---|---|
| Binary (tampered/authentic) | A+B1+C / single | **98.86%** | 99.41% |
| 5-class | A+B1+C / stacking | **98.52%** | 98.52% |

Evaluated on the **VEDTD** dataset (1,490 Vietnamese legal documents, 5-fold source-disjoint cross-validation).

---

## Installation

**Requirements:** Python 3.10+ · RAM ≥ 8 GB · ~1 GB disk for model cache

```bash
git clone https://github.com/Susois/mshf.git
cd mshf
python -m venv venv
source venv/bin/activate      # Linux/macOS
venv\Scripts\activate         # Windows
pip install -r requirements.txt
```

### Download the model

The trained model is distributed via **[GitHub Releases](https://github.com/Susois/mshf/releases/latest)**.

**Windows (PowerShell):**
```powershell
mkdir outputs\training
Invoke-WebRequest `
  -Uri "https://github.com/Susois/mshf/releases/latest/download/mshf_label.joblib" `
  -OutFile "outputs\training\mshf_label.joblib"
```

**Linux/macOS:**
```bash
mkdir -p outputs/training
wget -O outputs/training/mshf_label.joblib \
  https://github.com/Susois/mshf/releases/latest/download/mshf_label.joblib
```

---

## Quick Start

Place your PDFs in `sample_docs/` following this structure:

```
sample_docs/
├── 1.original/   ← reference document
├── 2.insert/     ← suspected document (same filename as original)
├── 3.delete/
├── 4.modify/
└── 5.layout/
```

Run detection:

```bash
python -m mshf.detect \
  --original  "sample_docs/1.original/document.pdf" \
  --candidate "sample_docs/2.insert/document.pdf" \
  --out-dir   "outputs/detect/result" \
  --mode full
```

> **`--mode full`** uses the complete A+B1+C pipeline — most accurate.  
> PhoBERT (~400 MB) and LayoutLMv3 are downloaded automatically on first run.  
> Use `--mode fast` (B1+C only, ~5 seconds, no internet) on low-RAM machines.

### Output

```
outputs/detect/result/
├── tampered_report.html   ← open in browser for visual report
├── report.json            ← full JSON result
└── page_000_highlighted.png
```

| Color | Meaning |
|---|---|
| 🔴 Red | Inserted line (not in original) |
| 🟠 Orange | Deleted line (removed from original) |
| 🟡 Yellow | Modified line (content changed) |

### Options

| Scenario | Parameter |
|---|---|
| Scanned PDF (no embedded text) | `--ocr paddleocr` |
| Low RAM / no internet | `--mode fast` |
| Custom model path | `--model path/to/mshf_label.joblib` |

---

## Python API

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

## Repository Structure

```
mshf/
├── mshf/
│   ├── core/          internal library (features, alignment, model, I/O)
│   ├── cli/           entry points (detect, train)
│   ├── pipeline/      data preparation
│   └── research/      evaluation & analysis
├── scripts/
│   ├── baseline/      Week 6 baselines (RF, PhoBERT, LayoutLMv3...)
│   ├── build_dataset/ VEDTD dataset generation
│   └── evaluation/    OCR and semantic evaluation
├── data/
│   └── hybrid_fusion_dataset.csv   Branch A dataset
├── results/           baseline comparison results
├── sample_docs/       place your PDFs here to run detection
├── outputs/           detection results and trained models
├── tests/
├── config.py
└── requirements.txt
```

---

## System Requirements

| Component | Minimum | Recommended |
|---|---|---|
| Python | 3.10 | 3.11+ |
| RAM | 8 GB | 16 GB |
| Disk | 1 GB | 2 GB |
| GPU | — | CUDA-compatible (speeds up Branch A) |

---

## License

This project is licensed under the [MIT License](LICENSE).  
See [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md) for third-party library attributions.
