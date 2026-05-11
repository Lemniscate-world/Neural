#!/usr/bin/env python3
"""Publish SESSION_SUMMARY.md content to a Google Doc.

This script supports two auth methods:
1) Service account JSON file (`--service-account-file`)
2) Service account JSON content (`--service-account-json`)

Typical usage:
  python scripts/publish_session_summary_to_gdocs.py \
      --source SESSION_SUMMARY.md \
      --doc-id <google-doc-id> \
      --service-account-file ./google-service-account.json \
      --mode append
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

SCOPES = ("https://www.googleapis.com/auth/documents",)
DOC_BASE_URL = "https://docs.google.com/document/d/{doc_id}/edit"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Publish SESSION_SUMMARY.md to Google Docs"
    )
    parser.add_argument(
        "--source",
        default="SESSION_SUMMARY.md",
        help="Path to markdown source file (default: SESSION_SUMMARY.md)",
    )
    parser.add_argument(
        "--doc-id",
        default=os.getenv("GOOGLE_DOC_ID", ""),
        help="Target Google Doc ID (or set GOOGLE_DOC_ID)",
    )
    parser.add_argument(
        "--service-account-file",
        default=os.getenv("GOOGLE_SERVICE_ACCOUNT_FILE", ""),
        help="Path to service account JSON file",
    )
    parser.add_argument(
        "--service-account-json",
        default=os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON", ""),
        help="Service account JSON content",
    )
    parser.add_argument(
        "--mode",
        choices=("append", "replace"),
        default="append",
        help="append: append latest summary block | replace: replace full doc body",
    )
    parser.add_argument(
        "--title",
        default="NeuralDBG Daily Work Summary",
        help="Document title if a new document is created",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print transformed output without sending to Google Docs",
    )
    return parser.parse_args()


def read_source_text(source_path: str) -> str:
    path = Path(source_path)
    if not path.exists():
        raise FileNotFoundError(f"Source file not found: {path}")
    return path.read_text(encoding="utf-8")


def build_payload_text(markdown_text: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    header = f"\n\n=== Session Sync ({stamp}) ===\n\n"
    return header + markdown_text.strip() + "\n"


def load_credentials(
    service_account_file: Optional[str],
    service_account_json: Optional[str],
):
    try:
        from google.oauth2 import service_account  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "Missing google-auth dependency. Install with: "
            "pip install google-auth google-api-python-client"
        ) from exc

    if service_account_file:
        path = Path(service_account_file)
        if not path.exists():
            raise FileNotFoundError(
                f"Service account file not found: {service_account_file}"
            )
        return service_account.Credentials.from_service_account_file(
            str(path), scopes=SCOPES
        )

    if service_account_json:
        info = json.loads(service_account_json)
        return service_account.Credentials.from_service_account_info(
            info, scopes=SCOPES
        )

    raise ValueError(
        "Missing credentials. Provide --service-account-file or "
        "--service-account-json (or matching env vars)."
    )


def build_docs_service(credentials):
    try:
        from googleapiclient.discovery import build  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "Missing google-api-python-client dependency. Install with: "
            "pip install google-auth google-api-python-client"
        ) from exc

    return build("docs", "v1", credentials=credentials)


def get_document_end_index(document: dict) -> int:
    body = document.get("body", {}).get("content", [])
    if not body:
        return 1
    end_index = body[-1].get("endIndex", 1)
    return max(1, int(end_index) - 1)


def create_document_if_needed(service, doc_id: str, title: str) -> str:
    if doc_id:
        return doc_id

    created = service.documents().create(body={"title": title}).execute()
    created_id = created.get("documentId")
    if not created_id:
        raise RuntimeError("Google Docs API did not return a documentId.")
    print(f"[INFO] Created new Google Doc: {DOC_BASE_URL.format(doc_id=created_id)}")
    return created_id


def update_document(service, doc_id: str, payload_text: str, mode: str) -> None:
    doc = service.documents().get(documentId=doc_id).execute()
    end_index = get_document_end_index(doc)

    requests = []
    if mode == "replace" and end_index > 1:
        requests.append(
            {
                "deleteContentRange": {
                    "range": {
                        "startIndex": 1,
                        "endIndex": end_index,
                    }
                }
            }
        )
        insert_index = 1
    else:
        insert_index = end_index

    requests.append(
        {
            "insertText": {
                "location": {"index": insert_index},
                "text": payload_text,
            }
        }
    )

    service.documents().batchUpdate(
        documentId=doc_id,
        body={"requests": requests},
    ).execute()


def main() -> int:
    args = parse_args()

    try:
        markdown_text = read_source_text(args.source)
        payload_text = build_payload_text(markdown_text)

        if args.dry_run:
            print(payload_text[:1200])
            return 0

        credentials = load_credentials(
            args.service_account_file,
            args.service_account_json,
        )
        service = build_docs_service(credentials)
        doc_id = create_document_if_needed(service, args.doc_id, args.title)
        update_document(service, doc_id, payload_text, args.mode)

        print(f"[OK] Google Doc updated: {DOC_BASE_URL.format(doc_id=doc_id)}")
        return 0

    except Exception as exc:  # noqa: BLE001
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
