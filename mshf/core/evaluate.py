"""Evaluation utilities with source-cluster confidence intervals and paired tests."""
from __future__ import annotations

import numpy as np
from scipy.stats import wilcoxon
from sklearn.metrics import (
    accuracy_score, average_precision_score, balanced_accuracy_score,
    brier_score_loss, confusion_matrix, f1_score, matthews_corrcoef,
    precision_recall_fscore_support, roc_auc_score,
)


def expected_calibration_error(y_true, probability, threshold: float = 0.5, n_bins: int = 10) -> float:
    probability = np.asarray(probability)
    correct = ((probability >= threshold).astype(int) == np.asarray(y_true)).astype(float)
    bins = np.linspace(0, 1, n_bins + 1); value = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (probability > lo) & (probability <= hi)
        if mask.any():
            value += mask.mean() * abs(correct[mask].mean() - probability[mask].mean())
    return float(value)


def cluster_bootstrap_ci(y_true, y_pred, groups, metric, n_boot: int = 1000, seed: int = 42):
    y_true, y_pred, groups = map(np.asarray, (y_true, y_pred, groups))
    unique = np.unique(groups); rng = np.random.default_rng(seed); scores = []
    for _ in range(n_boot):
        sampled = rng.choice(unique, len(unique), replace=True)
        idx = np.concatenate([np.flatnonzero(groups == g) for g in sampled])
        try: scores.append(metric(y_true[idx], y_pred[idx]))
        except ValueError: pass
    return tuple(float(v) for v in np.percentile(scores, [2.5, 97.5])) if scores else (float("nan"), float("nan"))


def evaluate_predictions(y_true, y_pred, labels, proba=None, groups=None, threshold=0.5):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    precision, recall, per_f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=range(len(labels)), zero_division=0)
    result = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "per_class": {lab: {"precision": float(p), "recall": float(r), "f1": float(f), "support": int(s)} for lab, p, r, f, s in zip(labels, precision, recall, per_f1, support)},
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=range(len(labels))).tolist(),
        "labels": labels, "threshold": float(threshold),
    }
    if groups is not None:
        result["macro_f1_ci95"] = list(cluster_bootstrap_ci(y_true, y_pred, groups, lambda a, b: f1_score(a, b, average="macro")))
        result["accuracy_ci95"] = list(cluster_bootstrap_ci(y_true, y_pred, groups, accuracy_score))
    if proba is not None:
        try:
            if len(labels) == 2:
                pos = proba[:, 1] if np.asarray(proba).ndim == 2 else np.asarray(proba)
                result.update(auroc=float(roc_auc_score(y_true, pos)), auprc=float(average_precision_score(y_true, pos)), brier=float(brier_score_loss(y_true, pos)), ece=expected_calibration_error(y_true, pos, threshold))
            else:
                result["auroc_ovr_macro"] = float(roc_auc_score(y_true, proba, multi_class="ovr", average="macro"))
        except ValueError as exc: result["probability_metric_error"] = str(exc)
    return result


def paired_tests(y_true, pred_a, pred_b, groups) -> dict:
    y_true, pred_a, pred_b, groups = map(np.asarray, (y_true, pred_a, pred_b, groups))
    scores_a, scores_b = [], []
    for group in np.unique(groups):
        mask = groups == group
        scores_a.append(float(np.mean(pred_a[mask] == y_true[mask])))
        scores_b.append(float(np.mean(pred_b[mask] == y_true[mask])))
    diff = np.asarray(scores_b) - np.asarray(scores_a)
    try: stat, p = wilcoxon(diff)
    except ValueError: stat, p = 0.0, 1.0
    effect = float(diff.mean() / diff.std(ddof=1)) if len(diff) > 1 and diff.std(ddof=1) else 0.0
    return {"wilcoxon_statistic": float(stat), "p_value": float(p), "mean_paired_difference": float(diff.mean()), "cohens_dz": effect}


def format_report(name, res):
    lines = [f"=== {name} ===", f"accuracy={res['accuracy']:.4f}", f"balanced_accuracy={res['balanced_accuracy']:.4f}", f"macro_f1={res['macro_f1']:.4f}", f"weighted_f1={res['weighted_f1']:.4f}", f"mcc={res['mcc']:.4f}"]
    for key in ("auroc", "auprc", "auroc_ovr_macro", "brier", "ece"):
        if key in res: lines.append(f"{key}={res[key]:.4f}")
    return "\n".join(lines)
