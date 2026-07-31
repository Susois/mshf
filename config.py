"""Portable MSHF paths and experiment constants.

Set MSHF_PROJECT_ROOT when data is outside the default sibling layout.
The repository does not contain raw PDFs, OCR, ground truth or derived outputs.
"""
from __future__ import annotations
import os
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent

# HYBRID_DATASET_CSV nằm trong repo tại data/
# Các đường dẫn dữ liệu thô (PDF, OCR, layout) không nằm trong repo —
# set biến môi trường tương ứng nếu cần train lại từ đầu.
HYBRID_DATASET_CSV = Path(os.environ.get("MSHF_HYBRID_DATASET", _REPO_ROOT / "data" / "hybrid_fusion_dataset.csv"))
PDF_ROOT           = Path(os.environ.get("MSHF_PDF_ROOT",        _REPO_ROOT / "data" / "vedtd" / "1.pdfs"))
OCR_TEXT_ROOT      = Path(os.environ.get("MSHF_OCR_ROOT",        _REPO_ROOT / "data" / "vedtd" / "3.ocr_output"))
GT_TEXT_ROOT       = Path(os.environ.get("MSHF_GT_ROOT",         _REPO_ROOT / "data" / "vedtd" / "2.ground_truth"))
LAYOUT_JSON_ROOT   = Path(os.environ.get("MSHF_LAYOUT_ROOT",     _REPO_ROOT / "data" / "vedtd" / "4.layout_ocr"))
OUTPUT_DIR         = Path(os.environ.get("MSHF_OUTPUT_DIR",      _REPO_ROOT / "outputs"))

ORIGINAL_CAT = "1.original"
CATEGORIES = ["1.original", "2.insert", "3.delete", "4.modify", "5.layout"]
CATEGORY_TO_LABEL = {"1.original":"original", "2.insert":"insert", "3.delete":"delete", "4.modify":"modify", "5.layout":"layout"}
DOC_FEATURE_COLS = ["cer", "wer", "mean_similarity", "min_similarity", "std_similarity", "ref_to_hyp_mean", "hyp_to_ref_mean", "layoutlmv3_cosine_similarity"]
CER_MODIFIED_THRESHOLD = 0.2
DELETE_LINES = 3
DELETE_PAGE_NUMBER = 2
