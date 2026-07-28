"""Normalize attack manifests and expose localization annotations."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import pandas as pd
import config

COLUMNS = ["sample_id","source_document_id","category","attack_type","attack_subtype","severity","page_id","original_text","tampered_text","affected_line_ids","affected_token_ids","bbox","template_id","seed","annotation_source"]


def normalize_manifest(source: Path, output: Path) -> pd.DataFrame:
    df=pd.read_csv(source)
    rename={"doc_id":"source_document_id","variant_id":"sample_id"}; df=df.rename(columns=rename)
    if "category" not in df:
        reverse={v:k for k,v in config.CATEGORY_TO_LABEL.items()}; df["category"]=df.get("attack_type","").map(reverse)
    defaults={"attack_subtype":"unknown","severity":"unknown","page_id":"","original_text":"","tampered_text":"","affected_line_ids":"[]","affected_token_ids":"[]","bbox":"[]","template_id":"unknown","seed":"","annotation_source":"legacy_manifest"}
    for col,value in defaults.items():
        if col not in df: df[col]=value
    for col in COLUMNS:
        if col not in df: df[col]=""
    output.parent.mkdir(parents=True,exist_ok=True); df[COLUMNS].to_csv(output,index=False,encoding="utf-8-sig"); return df[COLUMNS]


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--source",type=Path,default=config.PDF_ROOT/"manifest.csv"); ap.add_argument("--output",type=Path,default=config.OUTPUT_DIR/"manifest"/"attack_manifest.csv"); args=ap.parse_args(); df=normalize_manifest(args.source,args.output); print(f"{len(df)} rows -> {args.output}")
if __name__=="__main__": main()
