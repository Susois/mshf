# Security Policy

## Reporting

Do not open a public issue containing raw documents, OCR text, personal information, credentials, private repository URLs, or unpublished results. Report sensitive problems privately to the repository owner.

## Secrets

The repository must not contain API keys, access tokens, cloud credentials, private SSH keys, `.env` files, or authenticated dataset links. If a secret is committed, revoke it immediately and remove it from Git history; deleting it in a later commit is insufficient.

## Data

Raw and derived research data are excluded by `.gitignore`. Contributors must verify staged files before every push:

```bash
git status --short
git diff --cached --name-only
```
