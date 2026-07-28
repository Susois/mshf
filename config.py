"""Portable MSHF paths and experiment constants.

Set MSHF_PROJECT_ROOT when data is outside the default sibling layout.
The repository does not contain raw PDFs, OCR, ground truth or derived outputs.
"""
from __future__ import annotations
import os
from pathlib import Path

PROJECT_ROOT = Path(os.environ.get("MSHF_PROJECT_ROOT", Path(__file__).resolve().parent.parent)).expanduser().resolve()
HYBRID_DATASET_CSV = Path(os.environ.get("MSHF_HYBRID_DATASET", PROJECT_ROOT / "Tuan6" / "src" / "hybrid_fusion_dataset.csv"))
PDF_ROOT = Path(os.environ.get("MSHF_PDF_ROOT", PROJECT_ROOT / "Tuan1_2" / "VEDTD" / "1.pdfs"))
OCR_TEXT_ROOT = Path(os.environ.get("MSHF_OCR_ROOT", PROJECT_ROOT / "Tuan1_2" / "VEDTD" / "3.ocr_output"))
GT_TEXT_ROOT = Path(os.environ.get("MSHF_GT_ROOT", PROJECT_ROOT / "Tuan1_2" / "VEDTD" / "2.ground_truth"))
LAYOUT_JSON_ROOT = Path(os.environ.get("MSHF_LAYOUT_ROOT", PROJECT_ROOT / "Tuan5" / "4.layout_ocr"))
OUTPUT_DIR = Path(os.environ.get("MSHF_OUTPUT_DIR", Path(__file__).resolve().parent / "outputs"))

ORIGINAL_CAT = "1.original"
CATEGORIES = ["1.original", "2.insert", "3.delete", "4.modify", "5.layout"]
CATEGORY_TO_LABEL = {"1.original":"original", "2.insert":"insert", "3.delete":"delete", "4.modify":"modify", "5.layout":"layout"}
DOC_FEATURE_COLS = ["cer", "wer", "mean_similarity", "min_similarity", "std_similarity", "ref_to_hyp_mean", "hyp_to_ref_mean", "layoutlmv3_cosine_similarity"]
CER_MODIFIED_THRESHOLD = 0.2
DELETE_LINES = 3
DELETE_PAGE_NUMBER = 2
