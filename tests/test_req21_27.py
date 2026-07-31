"""Tests cho cac runner Yeu cau 21-26 (du lieu synthetic, khong can PDF that)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config


def _synthetic_dataset(n_sources: int = 12) -> pd.DataFrame:
    """Moi source co 1 original + 1 tampered, feature tach biet nhe theo nhan."""
    rng = np.random.default_rng(0)
    rows = []
    for i in range(n_sources):
        for cat, tampered in [("1.original", 0), ("2.insert", 1)]:
            rows.append({
                "sample_id": f"doc{i:03d}_{cat}",
                "source_document_id": f"doc{i:03d}",
                "category": cat,
                "is_tampered": tampered,
                "cer": float(rng.normal(0.05 + 0.3 * tampered, 0.01)),
                "wer": float(rng.normal(0.1 + 0.2 * tampered, 0.01)),
            })
    return pd.DataFrame(rows)


def _write_splits(df: pd.DataFrame, path: Path, n_folds: int = 3) -> Path:
    sources = sorted(df["source_document_id"].unique())
    splits = pd.DataFrame({
        "source_document_id": sources,
        "fold": [i % n_folds for i in range(len(sources))],
    })
    splits.to_csv(path, index=False)
    return path


class TestJoinB2:
    def test_join_saves_output(self, tmp_path, monkeypatch):
        monkeypatch.setattr(config, "OUTPUT_DIR", tmp_path)
        from mshf.pipeline import join_b2

        (tmp_path / "semantic").mkdir(parents=True)
        base = pd.DataFrame({
            "source_document_id": ["d1", "d1", "d2", "d2"],
            "category": ["1.original", "2.insert", "1.original", "2.insert"],
            "is_tampered": [0, 1, 0, 1],
        })
        b2 = pd.DataFrame({
            "source_document_id": ["d1", "d2"],
            "category": ["2.insert", "2.insert"],
            "b2_contradiction_mean": [0.5, 0.7],
            "b2_numeric_change_count": [1, 2],
        })
        base.to_csv(tmp_path / "enhanced_dataset.csv", index=False)
        b2.to_csv(tmp_path / "semantic" / "b2_features.csv", index=False)

        join_b2.main()

        out = pd.read_csv(tmp_path / "enhanced_dataset_b2.csv")
        assert len(out) == 4
        # original phai duoc fill 0, tampered giu gia tri B2
        orig = out[out.category == "1.original"]
        assert (orig[["b2_contradiction_mean", "b2_numeric_change_count"]] == 0).all().all()
        tamp = out[out.category == "2.insert"].sort_values("source_document_id")
        assert tamp["b2_contradiction_mean"].tolist() == [0.5, 0.7]


class TestRobustnessEval:
    def test_cross_validate_returns_metrics_and_predictions(self, tmp_path):
        from mshf.research.robustness_eval import cross_validate_robustness

        df = _synthetic_dataset()
        perturbed = df.copy()
        perturbed["cer"] += 0.05  # gia lap corruption
        split_file = _write_splits(df, tmp_path / "splits.csv")

        metrics, pred_df = cross_validate_robustness(df, perturbed, ["cer", "wer"], split_file)

        assert 0.0 <= metrics["macro_f1"] <= 1.0
        assert metrics["sample_count"] == len(df)
        assert len(pred_df) == len(df)
        assert {"y_true", "y_pred", "proba_tampered"} <= set(pred_df.columns)

    def test_perturb_features_preserves_shape_and_labels(self):
        from mshf.research.extract_control_features import perturb_features

        df = _synthetic_dataset()
        out = perturb_features(df, "noise", level=3, seed=42)
        assert len(out) == len(df)
        assert (out["is_tampered"] == df["is_tampered"]).all()
        assert out["sample_id"].str.endswith("_noise_3").all()


class TestUnseenEval:
    def test_run_holdout_no_source_leakage(self):
        from mshf.research.unseen_eval import run_holdout

        df = _synthetic_dataset(16)
        # generator A cho nua dau, B cho nua sau
        df["generator_id"] = ["genA" if i < 8 else "genB" for i in range(16) for _ in range(2)]

        metrics, pred_df, split_info = run_holdout(
            df, ["cer", "wer"], "is_tampered", "generator_id", "genB"
        )

        assert metrics is not None
        # khong co source nao xuat hien o ca train va test
        assert set(split_info["train_sources"]).isdisjoint(split_info["test_sources"])
        assert len(pred_df) == metrics["test_count"]


class TestCreateReport:
    def test_find_proba_column(self):
        from mshf.research.create_report import find_proba_column

        df = pd.DataFrame({"proba_0": [0.1], "proba_1": [0.9]})
        assert find_proba_column(df) == "proba_1"

    def test_case_studies_and_summary(self, tmp_path):
        from mshf.research.create_report import create_case_studies

        pred = pd.DataFrame({
            "sample_id": [f"s{i}" for i in range(8)],
            "source_document_id": [f"d{i}" for i in range(8)],
            "y_true": [0, 0, 1, 1, 0, 1, 0, 1],
            "y_pred": [0, 1, 1, 0, 0, 1, 0, 1],
            "proba_1": [0.1, 0.8, 0.9, 0.2, 0.3, 0.7, 0.4, 0.6],
        })
        out = tmp_path / "case_studies.csv"
        create_case_studies(pred, "proba_1", out, n_per_group=3)

        cases = pd.read_csv(out)
        assert set(cases["case_type"]) == {"TP", "TN", "FP", "FN"}
        # moi case truy nguoc duoc ve sample goc
        assert cases["source_document_id"].str.startswith("d").all()
