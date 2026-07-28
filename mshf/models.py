"""Bộ mô hình MSHF, giữ XGBoost làm lõi (đúng như đề tài gốc).

Ba biến thể:
  1. single      : một XGBoost trên toàn bộ feature đa mức (A+B+C).
  2. stacking     : base-learner mỗi nhánh (A / B / C) -> meta XGBoost.
  3. two_stage    : Stage-1 binary (authentic/tampered) -> Stage-2 5-class trên phần tampered.

Nếu không có xgboost -> fallback RandomForest (giữ nguyên tinh thần đề tài gốc).
"""
from __future__ import annotations

import numpy as np


def build_xgb(n_classes: int):
    """XGBoost với cấu hình gần đề tài gốc (n_estimators 300, depth 5)."""
    try:
        from xgboost import XGBClassifier

        params = dict(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
            tree_method="hist",
        )
        if n_classes == 2:
            params["eval_metric"] = "logloss"
        else:
            params["eval_metric"] = "mlogloss"
        return XGBClassifier(**params), "XGBoost"
    except ImportError:
        from sklearn.ensemble import RandomForestClassifier

        return RandomForestClassifier(
            n_estimators=300, max_depth=None, min_samples_leaf=2, random_state=42, n_jobs=-1
        ), "RandomForest"


class SingleModel:
    """Một XGBoost trên toàn bộ feature."""

    def __init__(self, n_classes: int):
        self.model, self.name = build_xgb(n_classes)

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)


class StackingModel:
    """Base-learner riêng cho từng nhánh feature -> meta-learner XGBoost.

    branch_slices: dict tên_nhánh -> list chỉ số cột feature của nhánh đó.
    """

    def __init__(self, n_classes: int, branch_slices: dict[str, list[int]]):
        self.n_classes = n_classes
        self.branch_slices = branch_slices
        self.bases: dict[str, object] = {}
        self.meta, self.name = build_xgb(n_classes)
        self.name = f"Stacking({self.name})"

    def _meta_features(self, X, fit: bool, y=None):
        cols = []
        for branch, idx in self.branch_slices.items():
            if fit:
                base, _ = build_xgb(self.n_classes)
                base.fit(X[:, idx], y)
                self.bases[branch] = base
            proba = self.bases[branch].predict_proba(X[:, idx])
            cols.append(proba)
        return np.hstack(cols)

    def fit_oof(self, X, y, groups, inner_folds: int = 4):
        """Fit leakage-free branch stacking using group-disjoint OOF meta-features."""
        from sklearn.model_selection import GroupKFold
        groups = np.asarray(groups)
        unique = np.unique(groups)
        n_splits = min(inner_folds, len(unique))
        meta_X = np.zeros((len(y), self.n_classes * len(self.branch_slices)), dtype=float)
        splitter = GroupKFold(n_splits=n_splits)
        for tr, va in splitter.split(X, y, groups):
            offset = 0
            for branch, idx in self.branch_slices.items():
                base, _ = build_xgb(self.n_classes)
                base.fit(X[tr][:, idx], y[tr])
                proba = base.predict_proba(X[va][:, idx])
                meta_X[va, offset:offset + proba.shape[1]] = proba
                offset += self.n_classes
        self.meta.fit(meta_X, y)
        for branch, idx in self.branch_slices.items():
            base, _ = build_xgb(self.n_classes)
            base.fit(X[:, idx], y)
            self.bases[branch] = base
        return self


    def predict(self, X):
        return self.meta.predict(self._meta_features(X, fit=False))

    def predict_proba(self, X):
        return self.meta.predict_proba(self._meta_features(X, fit=False))


class TwoStageModel:
    """Stage-1 binary tampered/authentic; Stage-2 phân loại 4 kiểu tấn công.

    Dự đoán: nếu Stage-1 nói authentic -> nhãn original; ngược lại dùng Stage-2.
    Yêu cầu biết chỉ số nhãn 'original' trong bảng nhãn multi-class.
    """

    def __init__(self, original_idx: int, tampered_labels: list[int]):
        self.original_idx = original_idx
        self.tampered_labels = tampered_labels
        self.stage1, _ = build_xgb(2)
        self.stage2, self.name = build_xgb(len(tampered_labels))
        self.name = f"TwoStage({self.name})"
        self._t2_map = {lab: i for i, lab in enumerate(sorted(tampered_labels))}
        self._t2_inv = {i: lab for lab, i in self._t2_map.items()}

    def fit(self, X, y):
        is_tampered = (y != self.original_idx).astype(int)
        self.stage1.fit(X, is_tampered)
        mask = is_tampered == 1
        y2 = np.array([self._t2_map[v] for v in y[mask]])
        self.stage2.fit(X[mask], y2)
        return self

    def predict(self, X):
        s1 = self.stage1.predict(X)
        out = np.full(len(X), self.original_idx)
        mask = s1 == 1
        if mask.any():
            s2 = self.stage2.predict(X[mask])
            out[mask] = np.array([self._t2_inv[v] for v in s2])
        return out
