# Repository Release Checklist

## Before the first private push

- [ ] Repository visibility is **Private**.
- [ ] Raw/tampered PDFs are not staged.
- [ ] OCR, ground-truth text, line crops and sensitive manifests are not staged.
- [ ] `outputs/`, checkpoints and unpublished result tables are not staged.
- [ ] Word/PDF manuscripts and temporary Office files are not staged.
- [ ] `.env`, API keys, tokens and cloud credentials are not staged.
- [ ] Absolute local paths and personal identifiers have been removed from public source/docs.
- [ ] `git diff --cached --name-only` contains only intended source and documentation.
- [ ] Tests pass.

## Before making the repository public

- [ ] Target venue permits public code/preprints before review.
- [ ] Double-blind policy has been checked.
- [ ] Patent/IP review is complete if relevant.
- [ ] Code ownership and contributor authorship are agreed.
- [ ] A code license has been approved by the owner/institution.
- [ ] Third-party package and checkpoint licenses are recorded.
- [ ] Data Availability Statement is accurate.
- [ ] Git history contains no deleted secrets or research data.
- [ ] README, commit metadata and release assets do not identify authors if anonymous review is required.

## Recommended workflow

1. Develop and collaborate in a private repository.
2. Tag the exact submission commit privately.
3. For double-blind review, prepare an anonymized archive or venue-approved anonymous repository.
4. Make the canonical repository public only when publication/IP/data policies allow it.
