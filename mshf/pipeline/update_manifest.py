from __future__ import annotations

import hashlib
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import config


SUBTYPE_MAP = {
    "original": ["none"],
    "insert": ["character", "token", "phrase"],
    "delete": ["character", "token", "line"],
    "modify": ["lexical", "numeric", "entity"],
    "layout": ["spacing", "font", "margin"],
}

SEVERITY_OPTIONS = ["low", "medium", "high"]


def get_template(doc_id: str) -> str:
    """Trích template_id từ tên document (NQ-HDND, QH16, UBTVQH16...)."""
    match = re.search(r"\d{4}_(.+?)_\d+$", str(doc_id))
    return match.group(1) if match else "unknown"


def deterministic_choice(options: list, key: str) -> str:
    """Chọn deterministic từ danh sách dựa trên hash của key."""
    h = int(hashlib.md5(key.encode("utf-8")).hexdigest(), 16)
    return options[h % len(options)]


def main():
    manifest_path = config.OUTPUT_DIR / "manifest" / "attack_manifest.csv"

    if not manifest_path.exists():
        print(f"Manifest not found: {manifest_path}")
        return

    m = pd.read_csv(manifest_path)
    print(f"Loaded manifest: {len(m)} rows, {m['source_document_id'].nunique()} sources")
    print(f"  Before - severity: {m['severity'].value_counts().to_dict()}")

    # 1. Template ID
    m["template_id"] = m["source_document_id"].apply(get_template)

    # 2. Attack subtype
    for idx, row in m.iterrows():
        at = str(row["attack_type"])
        if at == "original":
            m.at[idx, "attack_subtype"] = "none"
            m.at[idx, "severity"] = "none"
            m.at[idx, "annotation_source"] = "none"
        else:
            doc_id = str(row["source_document_id"])

            # Subtype
            options = SUBTYPE_MAP.get(at, ["unknown"])
            m.at[idx, "attack_subtype"] = deterministic_choice(options, doc_id)

            # Severity
            m.at[idx, "severity"] = deterministic_choice(
                SEVERITY_OPTIONS, doc_id + at
            )

            # Annotation source
            m.at[idx, "annotation_source"] = "rule_based_hash"

    # Save
    m.to_csv(manifest_path, index=False, encoding="utf-8-sig")

    print(f"\n  After - severity: {m['severity'].value_counts().to_dict()}")
    print(f"  After - subtype:  {m['attack_subtype'].value_counts().to_dict()}")
    print(f"  After - template: {m['template_id'].value_counts().to_dict()}")
