# Google Docs Sync Setup

This guide automates publishing `SESSION_SUMMARY.md` to a Google Doc so daily summaries are no longer manual.

## Option A: Local Script

1. Install automation dependencies:
```bash
pip install -e .[automation]
```

2. Create a Google Cloud service account:
- Enable **Google Docs API** in your Google Cloud project.
- Create a service account and download its JSON key.
- Share your target Google Doc with the service account email (Editor role).

3. Set environment variables:
```bash
export GOOGLE_DOC_ID="<your-google-doc-id>"
export GOOGLE_SERVICE_ACCOUNT_FILE="./google-service-account.json"
```

PowerShell:
```powershell
$env:GOOGLE_DOC_ID="<your-google-doc-id>"
$env:GOOGLE_SERVICE_ACCOUNT_FILE="C:\\path\\to\\google-service-account.json"
```

4. Run sync:
```bash
python scripts/publish_session_summary_to_gdocs.py --source SESSION_SUMMARY.md --mode append
```

Notes:
- `--mode append` adds a timestamped block each run.
- `--mode replace` replaces the document body with the latest summary.
- If `--doc-id` is omitted, a new Google Doc is created.

## Option B: GitHub Actions Daily Sync

Workflow file:
- `.github/workflows/publish-summary-to-google-docs.yml`

Add repository secrets:
- `GOOGLE_DOC_ID`
- `GOOGLE_SERVICE_ACCOUNT_JSON` (full JSON content, not file path)

Then:
- Run manually from Actions (`workflow_dispatch`), or
- Let schedule run weekdays at `22:30 UTC`.

## Security

- Never commit service account JSON keys.
- `.gitignore` includes:
  - `google-service-account.json`
  - `*.service-account.json`
