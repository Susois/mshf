"""GPU/Colab Vietnamese NLI inference for aligned MSHF line pairs."""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--pairs",type=Path,required=True); ap.add_argument("--output",type=Path,required=True); ap.add_argument("--model",default="joeddav/xlm-roberta-large-xnli"); ap.add_argument("--batch-size",type=int,default=16); args=ap.parse_args()
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    df=pd.read_json(args.pairs,lines=True); tok=AutoTokenizer.from_pretrained(args.model); model=AutoModelForSequenceClassification.from_pretrained(args.model).eval(); device=torch.device("cuda" if torch.cuda.is_available() else "cpu"); model.to(device)
    labels={v.lower():i for i,v in model.config.id2label.items()}; required=["contradiction","entailment","neutral"]
    if not all(x in labels for x in required): raise ValueError(f"Model labels must include {required}: {model.config.id2label}")
    rows=[]
    with torch.no_grad():
        for start in range(0,len(df),args.batch_size):
            part=df.iloc[start:start+args.batch_size]; enc=tok(part.ref_line.tolist(),part.cand_line.tolist(),padding=True,truncation=True,max_length=256,return_tensors="pt").to(device); probs=model(**enc).logits.softmax(-1).cpu().numpy()
            for (_,r),prob in zip(part.iterrows(),probs): rows.append({"pair_id":r.pair_id,**{lab:float(prob[labels[lab]]) for lab in required},"model":args.model})
    args.output.parent.mkdir(parents=True,exist_ok=True); pd.DataFrame(rows).to_csv(args.output,index=False,encoding="utf-8-sig"); print(f"{len(rows)} pairs -> {args.output} ({device})")
if __name__=="__main__": main()
