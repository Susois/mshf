"""Prepare aligned line pairs and aggregate Semantic Critical Change (B2) features."""
from __future__ import annotations
import argparse, json, re
from pathlib import Path
import numpy as np, pandas as pd
import config
from . import io_utils
from .line_align import match_lines

B2_FEATURE_COLS=["b2_contradiction_mean","b2_contradiction_max","b2_contradiction_p90","b2_contradiction_high_count","b2_entity_change_count","b2_numeric_change_count","b2_unit_change_count","b2_negation_change_count","b2_obligation_change_count","b2_logic_change_count"]
NUMBER=re.compile(r"\d+(?:[.,/]\d+)*")
UNITS=("đồng","%","kg","km","m2","triệu","tỷ")
NEG=("không","chưa","cấm")
OBLIGATION=("phải","được phép","có nghĩa vụ","có quyền","bắt buộc")
LOGIC=("và","hoặc","trước","sau","tối đa","tối thiểu")


def prepare_pairs(output: Path, max_docs=0):
    rows=[]; docs=io_utils.discover_doc_ids(); docs=docs[:max_docs] if max_docs else docs
    for doc in docs:
        for cat in config.CATEGORIES[1:]:
            ref=io_utils.read_text_lines(io_utils.ocr_text_path(config.ORIGINAL_CAT,doc)); cand=io_utils.read_text_lines(io_utils.ocr_text_path(cat,doc))
            for i,p in enumerate(match_lines(io_utils.as_line_dicts(ref),io_utils.as_line_dicts(cand))):
                if p["type"]=="match" and p["cer"]>0:
                    rows.append({"pair_id":f"{doc}|{cat}|{i}","source_document_id":doc,"category":cat,"ref_line":p["orig_line"]["text"],"cand_line":p["cand_line"]["text"],"cer":p["cer"],"ocr_confidence":1.0})
    output.parent.mkdir(parents=True,exist_ok=True); pd.DataFrame(rows).to_json(output,orient="records",lines=True,force_ascii=False); return len(rows)


def changed_terms(a,b,terms): return sum(a.count(t)!=b.count(t) for t in terms)

def aggregate(pairs: Path, nli: Path, output: Path):
    df=pd.read_json(pairs,lines=True); scores=pd.read_csv(nli); df=df.merge(scores[["pair_id","contradiction","entailment","neutral"]],on="pair_id",how="left",validate="one_to_one")
    if df.contradiction.isna().any(): raise ValueError("NLI output does not cover all line pairs")
    feature_rows=[]
    for (doc,cat),g in df.groupby(["source_document_id","category"]):
        weights=g.ocr_confidence.clip(lower=.01).to_numpy(); c=g.contradiction.to_numpy(); entity=numeric=unit=neg=obligation=logic=0
        for _,r in g.iterrows():
            a,b=str(r.ref_line).lower(),str(r.cand_line).lower(); numeric+=NUMBER.findall(a)!=NUMBER.findall(b); unit+=changed_terms(a,b,UNITS); neg+=changed_terms(a,b,NEG); obligation+=changed_terms(a,b,OBLIGATION); logic+=changed_terms(a,b,LOGIC)
            entity+=int(any(x.isupper() and len(x)>1 for x in set(str(r.ref_line).split())^set(str(r.cand_line).split())))
        feature_rows.append({"source_document_id":doc,"category":cat,"b2_contradiction_mean":float(np.average(c,weights=weights)),"b2_contradiction_max":float(c.max()),"b2_contradiction_p90":float(np.percentile(c,90)),"b2_contradiction_high_count":int((c>=.7).sum()),"b2_entity_change_count":entity,"b2_numeric_change_count":numeric,"b2_unit_change_count":unit,"b2_negation_change_count":neg,"b2_obligation_change_count":obligation,"b2_logic_change_count":logic})
    output.parent.mkdir(parents=True,exist_ok=True); pd.DataFrame(feature_rows).to_csv(output,index=False,encoding="utf-8-sig")


def main():
    ap=argparse.ArgumentParser(); sub=ap.add_subparsers(dest="cmd",required=True); a=sub.add_parser("prepare"); a.add_argument("--output",type=Path,default=config.OUTPUT_DIR/"semantic"/"line_pairs.jsonl"); a.add_argument("--max-docs",type=int,default=0); b=sub.add_parser("aggregate"); b.add_argument("--pairs",type=Path,required=True); b.add_argument("--nli",type=Path,required=True); b.add_argument("--output",type=Path,default=config.OUTPUT_DIR/"semantic"/"b2_features.csv"); args=ap.parse_args(); print(prepare_pairs(args.output,args.max_docs) if args.cmd=="prepare" else aggregate(args.pairs,args.nli,args.output))
if __name__=="__main__": main()
