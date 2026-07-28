"""Leakage-free training, calibration, fixed splits and publication artifacts."""
from __future__ import annotations

import argparse, json, platform, sys
from pathlib import Path
import joblib, numpy as np, pandas as pd, sklearn
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from mshf.evaluate import evaluate_predictions, format_report, paired_tests
from mshf.geometric_features import GEOMETRIC_FEATURE_COLS
from mshf.line_features import LINE_FEATURE_COLS
from mshf.models import SingleModel, StackingModel, TwoStageModel


def branches(cols):
    return {"A": [cols.index(c) for c in config.DOC_FEATURE_COLS if c in cols], "B1": [cols.index(c) for c in LINE_FEATURE_COLS if c in cols], "C": [cols.index(c) for c in GEOMETRIC_FEATURE_COLS if c in cols]}


def choose_threshold(y, probability):
    candidates = np.linspace(0.1, 0.9, 81)
    scores = [f1_score(y, probability >= t, average="macro") for t in candidates]
    return float(candidates[int(np.argmax(scores))])


def fit_model(kind, X, y, groups, cols, labels):
    if kind == "stacking":
        return StackingModel(len(labels), {k: v for k, v in branches(cols).items() if v}).fit_oof(X, y, groups)
    if kind == "two_stage":
        original = labels.index("original")
        return TwoStageModel(original, [i for i in range(len(labels)) if i != original]).fit(X, y)
    return SingleModel(len(labels)).fit(X, y)


def split_iterator(df, split_file, folds):
    if split_file and split_file.exists():
        mapping = pd.read_csv(split_file).set_index("source_document_id")["fold"]
        assigned = df.source_document_id.astype(str).map(mapping)
        if assigned.isna().any(): raise ValueError("Split file does not cover every source_document_id")
        for fold in sorted(assigned.unique()):
            yield int(fold), np.flatnonzero(assigned.to_numpy() != fold), np.flatnonzero(assigned.to_numpy() == fold)
    else:
        groups = df.source_document_id.astype(str).to_numpy()
        for fold, (tr, te) in enumerate(GroupKFold(min(folds, len(np.unique(groups)))).split(df, groups=groups), 1): yield fold, tr, te


def cross_validate(df, target, cols, kind, split_file=None, folds=5):
    X = df[cols].replace([np.inf, -np.inf], np.nan)
    if X.isna().any().any(): raise ValueError(f"Missing/non-finite features: {X.columns[X.isna().any()].tolist()}")
    X = X.to_numpy(float); le = LabelEncoder(); y = le.fit_transform(df[target].astype(str)); labels = list(le.classes_)
    groups = df.source_document_id.astype(str).to_numpy(); pred = np.empty_like(y); proba = np.zeros((len(y), len(labels))); rows=[]
    for fold, tr, te in split_iterator(df, split_file, folds):
        model = fit_model(kind, X[tr], y[tr], groups[tr], cols, labels)
        fold_proba = model.predict_proba(X[te]) if hasattr(model, "predict_proba") else None
        threshold = 0.5
        if len(labels) == 2 and fold_proba is not None:
            inner = GroupKFold(min(4, len(np.unique(groups[tr])))); inner_prob=np.zeros(len(tr))
            for itr, iva in inner.split(X[tr], y[tr], groups[tr]):
                inner_model = fit_model(kind, X[tr][itr], y[tr][itr], groups[tr][itr], cols, labels)
                inner_prob[iva] = inner_model.predict_proba(X[tr][iva])[:, 1]
            threshold = choose_threshold(y[tr], inner_prob)
            pred[te] = (fold_proba[:, 1] >= threshold).astype(int)
        else: pred[te] = model.predict(X[te])
        if fold_proba is not None: proba[te] = fold_proba
        rows.append({"fold": fold, "train": len(tr), "test": len(te), "threshold": threshold, "macro_f1": f1_score(y[te], pred[te], average="macro"), "balanced_accuracy": balanced_accuracy_score(y[te], pred[te])})
    res = evaluate_predictions(y, pred, labels, proba, groups)
    res.update(folds=rows, protocol="fixed source-disjoint folds" if split_file else "GroupKFold", features=cols, model_kind=kind)
    predictions = pd.DataFrame({"sample_id": df.sample_id if "sample_id" in df else df.index, "source_document_id": groups, "target": target, "y_true": y, "y_pred": pred})
    for i, label in enumerate(labels): predictions[f"proba_{label}"] = proba[:, i]
    return res, predictions, labels


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--dataset",type=Path,default=config.OUTPUT_DIR/"enhanced_dataset.csv"); ap.add_argument("--out",type=Path,default=config.OUTPUT_DIR/"training"); ap.add_argument("--splits",type=Path,default=config.OUTPUT_DIR/"audit"/"source_splits.csv"); ap.add_argument("--folds",type=int,default=5); ap.add_argument("--task",choices=["binary","multi","both"],default="both"); ap.add_argument("--models",nargs="+",choices=["single","stacking","two_stage"],default=["single","stacking","two_stage"]); args=ap.parse_args()
    df=pd.read_csv(args.dataset); args.out.mkdir(parents=True,exist_ok=True)
    A=config.DOC_FEATURE_COLS; B=LINE_FEATURE_COLS; C=GEOMETRIC_FEATURE_COLS
    ablations={"A":A,"A+B1":A+B,"A+C":A+C,"B1+C":B+C,"A+B1+C":A+B+C}; targets=["is_tampered","label"] if args.task=="both" else (["is_tampered"] if args.task=="binary" else ["label"])
    results=[]; predictions={}; reports=[]
    for target in targets:
        for ablation, cols in ablations.items():
            kinds=["single"] if ablation!="A+B1+C" else list(dict.fromkeys(args.models))
            for kind in kinds:
                if kind=="two_stage" and target!="label": continue
                name=f"{target}/{ablation}/{kind}"; res,pred,labels=cross_validate(df,target,cols,kind,args.splits,args.folds); res.update(name=name,target=target,ablation=ablation); results.append(res); predictions[name]=pred; reports.append(format_report(name,res)); pred.to_csv(args.out/f"predictions_{target}_{ablation.replace('+','_')}_{kind}.csv",index=False,encoding="utf-8-sig")
    tests=[]
    for target in targets:
        base=f"{target}/A/single"; full=f"{target}/A+B1+C/single"
        if base in predictions and full in predictions:
            a=predictions[base]; b=predictions[full]; tests.append({"target":target,"comparison":"A_vs_full",**paired_tests(a.y_true,a.y_pred,b.y_pred,a.source_document_id)})
    (args.out/"training_summary.json").write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding="utf-8"); (args.out/"statistical_tests.json").write_text(json.dumps(tests,ensure_ascii=False,indent=2),encoding="utf-8"); (args.out/"report.txt").write_text("\n\n".join(reports),encoding="utf-8")
    pd.DataFrame([{k:r.get(k) for k in ["name","target","ablation","model_kind","accuracy","balanced_accuracy","macro_f1","weighted_f1","mcc","auroc","auprc","auroc_ovr_macro","brier","ece"]} for r in results]).to_csv(args.out/"comparison.csv",index=False,encoding="utf-8-sig")
    run={"seed":42,"python":platform.python_version(),"sklearn":sklearn.__version__,"dataset":str(args.dataset),"splits":str(args.splits)}; (args.out/"run_config.json").write_text(json.dumps(run,indent=2),encoding="utf-8")
    full=A+B+C
    for target in targets:
        le=LabelEncoder(); y=le.fit_transform(df[target].astype(str)); model=SingleModel(len(le.classes_)).fit(df[full].to_numpy(float),y); joblib.dump({"model":model,"labels":list(le.classes_),"features":full,"seed":42},args.out/f"mshf_{target}.joblib")
    print(f"DONE -> {args.out}")

if __name__=="__main__": main()
