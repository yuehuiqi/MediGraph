# Security Policy

## Sensitive information

- Never commit `.env`, API keys, patient identifiers, credentials or private endpoints.
- Keep `.env`, virtual environments, caches and transient logs outside Git; the
  repository `.gitignore` defines these exclusions.
- Run `python scripts/check_release.py` before every upload.

## Medical data

The `pii_redact` operator detects common identifiers, but regex redaction is not a
formal de-identification guarantee. Real clinical data requires institutional
approval, access controls and human review.

## SQL and file handling

NL2SQL uses a read-only SQLite connection, an authorizer, a timeout and a row limit.
The document loader enforces a size limit and an explicit format allow-list.

Please report vulnerabilities privately to the project maintainers rather than
opening an issue containing secrets or patient information.
