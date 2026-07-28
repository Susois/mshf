"""Dataset inventory, quality control and canonical source-disjoint splits."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupKFold

import config

CATEGORIES = list(config.CATEGORIES)
LABELS = [config.CATEGORY_TO_LABEL[c] for c in CATEGORIES]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _file(category: str, doc_id: str, kind: str) -> Path:
    if kind == "pdf":
        return config.PDF_ROOT / category / f"{doc_id}.pdf"
    if kind == "ocr":
        return config.OCR_TEXT_ROOT / category / f"{doc_id}.txt"
    if kind == "gt":
        return config.GT_TEXT_ROOT / category / f"{doc_id}.txt"
    if kind == "layout":
        return config.LAYOUT_JSON_ROOT / category / f"{doc_id}.json"
    raise ValueError(kind)


def expected_doc_ids() -> list[str]:
    csv_path = config.HYBRID_DATASET_CSV
    if not csv_path.exists():
        return []
    df = pd.read_csv(csv_path)
    key = "source_document_id" if "source_document_id" in df else "key"
    return sorted(df[key].astype(str).unique())


def build_inventory() -> tuple[pd.DataFrame, dict]:
    ids = expected_doc_ids()
    rows = []
    for doc_id in ids:
        for category in CATEGORIES:
            pdf = _file(category, doc_id, "pdf")
            ocr = _file(category, doc_id, "ocr")
            gt = _file(category, doc_id, "gt")
            layout = _file(category, doc_id, "layout")
            row = {
                "sample_id": f"{doc_id}_{config.CATEGORY_TO_LABEL[category]}",
                "source_document_id": doc_id,
                "category": category,
                "label": config.CATEGORY_TO_LABEL[category],
                "is_tampered": int(category != config.ORIGINAL_CAT),
                "pdf_path": str(pdf), "ocr_path": str(ocr),
                "gt_path": str(gt), "layout_path": str(layout),
                "pdf_exists": pdf.exists(), "ocr_exists": ocr.exists(),
                "gt_exists": gt.exists(), "layout_exists": layout.exists(),
                "pdf_sha256": sha256(pdf) if pdf.exists() else "",
                "ocr_chars": len(ocr.read_text(encoding="utf-8", errors="replace")) if ocr.exists() else 0,
            }
            rows.append(row)
    inv = pd.DataFrame(rows)
    checks = {
        "source_documents": int(inv.source_document_id.nunique()) if not inv.empty else 0,
        "categories": {c: int((inv.category == c).sum()) for c in CATEGORIES} if not inv.empty else {},
        "samples": int(len(inv)),
        "expected_samples": int(len(ids) * len(CATEGORIES)),
        "missing_files": int((~inv[["pdf_exists", "ocr_exists", "gt_exists", "layout_exists"]]).sum().sum()) if not inv.empty else 0,
        "duplicate_pdf_hashes": int(inv.loc[inv.pdf_sha256.ne(""), "pdf_sha256"].duplicated(keep=False).sum()) if not inv.empty else 0,
        "empty_ocr": int((inv.ocr_chars == 0).sum()) if not inv.empty else 0,
    }
    checks["ok"] = bool(checks["samples"] == checks["expected_samples"] and checks["missing_files"] == 0 and checks["empty_ocr"] == 0)
    return inv, checks


def make_splits(inventory: pd.DataFrame, folds: int = 5, seed: int = 42) -> pd.DataFrame:
    groups = inventory[["source_document_id"]].drop_duplicates().sort_values("source_document_id").reset_index(drop=True)
    splitter = GroupKFold(n_splits=min(folds, len(groups)))
    out = groups.copy()
    out["fold"] = -1
    dummy = out["source_document_id"].to_numpy()
    for fold, (_, te) in enumerate(splitter.split(dummy, groups=dummy), 1):
        out.loc[te, "fold"] = fold
    out["seed"] = seed
    return inventory.merge(out, on="source_document_id", how="left")


def audit(out_dir: Path, folds: int = 5, strict: bool = False) -> tuple[pd.DataFrame, dict]:
    out_dir.mkdir(parents=True, exist_ok=True)
    inv, checks = build_inventory()
    inv = make_splits(inv, folds=folds)
    inv.to_csv(out_dir / "dataset_inventory.csv", index=False, encoding="utf-8-sig")
    (out_dir / "audit_summary.json").write_text(json.dumps(checks, ensure_ascii=False, indent=2), encoding="utf-8")
    report = ["MSHF DATASET AUDIT", json.dumps(checks, ensure_ascii=False, indent=2)]
    (out_dir / "audit_report.txt").write_text("\n\n".join(report), encoding="utf-8")
    inv[["source_document_id", "fold", "seed"]].drop_duplicates().to_csv(out_dir / "source_splits.csv", index=False, encoding="utf-8-sig")
    if strict and not checks["ok"]:
        raise RuntimeError(f"Dataset audit failed: {checks}")
    return inv, checks


def main() -> None:
    ap = argparse.ArgumentParser(description="Audit MSHF dataset and create fixed GroupKFold splits")
    ap.add_argument("--out", type=Path, default=config.OUTPUT_DIR / "audit")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--strict", action="store_true")
    args = ap.parse_args()
    _, checks = audit(args.out, args.folds, args.strict)
    print(json.dumps(checks, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
