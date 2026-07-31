from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config


def main():
    # Load datasets
    base = pd.read_csv(config.OUTPUT_DIR / 'enhanced_dataset.csv')
    b2 = pd.read_csv(config.OUTPUT_DIR / 'semantic' / 'b2_features.csv')

    print(f'Base dataset shape: {base.shape}')
    print(f'B2 features shape: {b2.shape}')

    # Join theo source_document_id + category
    result = base.merge(
        b2,
        on=['source_document_id', 'category'],
        how='left',
        validate='one_to_one'
    )

    print(f'After merge: {result.shape}')

    # Fill original documents với 0 (không có semantic changes)
    b2_cols = [c for c in b2.columns if c.startswith('b2_')]
    result.loc[result.category == '1.original', b2_cols] = 0

    # Kiểm tra missing — một số tampered docs có thể không có changed lines
    missing_mask = result[b2_cols].isna().any(axis=1)
    missing_count = missing_mask.sum()
    if missing_count > 0:
        missing_docs = result.loc[missing_mask, ['source_document_id', 'category']].values.tolist()
        print(f'WARNING: {missing_count} tampered docs thieu B2 features (khong co changed lines)')
        print(f'  Fill 0 cho cac docs nay (khong co semantic change detected)')
        for doc_id, cat in missing_docs[:10]:
            print(f'    {doc_id} / {cat}')
        if missing_count > 10:
            print(f'    ... va {missing_count - 10} docs khac')
        # Fill 0 cho tampered docs thiếu B2 (no contradiction detected)
        result[b2_cols] = result[b2_cols].fillna(0)

    print(f'B2 features joined: {len(b2_cols)} columns')

    # Kiểm tra cuối: không còn missing sau khi fill
    remaining_missing = int(result[b2_cols].isna().sum().sum())
    if remaining_missing > 0:
        raise ValueError(f'Missing B2 features sau khi fill: {remaining_missing} cells')

    # Lưu dataset đã join
    output = config.OUTPUT_DIR / 'enhanced_dataset_b2.csv'
    result.to_csv(output, index=False, encoding='utf-8-sig')
    print(f'Saved to: {output} ({len(result)} rows, {len(result.columns)} cols)')


if __name__ == '__main__':
    main()
