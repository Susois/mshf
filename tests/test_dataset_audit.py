from pathlib import Path
import pandas as pd

from mshf.dataset_audit import build_inventory, make_splits


def test_real_inventory_has_expected_structure():
    inventory, checks = build_inventory()
    assert checks["source_documents"] == 298
    assert checks["samples"] == 1490
    assert all(v == 298 for v in checks["categories"].values())
    assert checks["missing_files"] == 0


def test_source_split_never_leaks():
    inventory, _ = build_inventory()
    split = make_splits(inventory, folds=5)
    assert split.groupby("source_document_id").fold.nunique().max() == 1
    assert split.fold.nunique() == 5
