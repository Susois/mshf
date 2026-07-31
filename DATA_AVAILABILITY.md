# Data Availability and Responsible Release

This repository contains source code only. It does **not** distribute raw PDFs, tampered PDFs, OCR/ground-truth text, line crops, manifests containing document content, trained checkpoints, or experiment outputs.

## Dataset

The study uses 298 source documents and four controlled manipulation variants per source (1,490 PDF samples in the complete five-class structure). Access to the underlying documents is subject to copyright, privacy, institutional, and source-specific restrictions.

Before any dataset release:

1. verify redistribution rights for every source document;
2. remove personal identifiers, signatures, seals, and restricted content;
3. document provenance and consent/legal basis where applicable;
4. distinguish direct annotations from OCR-derived ground truth;
5. obtain any required institutional approval.

Where raw documents cannot be redistributed, a release may provide anonymized metadata, fixed split identifiers, attack-generation code, schemas, aggregate features that cannot reconstruct content, and evaluation scripts.

## Third-party models and software

Pretrained models and dependencies retain their original licenses. Users must review the license and model card for each selected checkpoint before use or redistribution. See `THIRD_PARTY_LICENSES.md`.

## Publication

Reuse of this code or dataset in later publications must disclose the prior study, clearly identify the new contribution, and comply with the target venue's prior-publication, preprint, software-release, and double-blind policies.
