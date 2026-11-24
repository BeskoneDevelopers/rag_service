"""Reindex existing DOCX/Markdown files for an explicit company without re-uploading."""

import argparse
from pathlib import Path

from sqlalchemy import select

from app.models import Document
from app.tasks import process_document
from app.worker_db import sessionlocal


def reindex(company: str, filename: str | None, apply: bool) -> list[dict]:
    results = []
    # One short transaction per document; indexing itself keeps the row locked.
    with sessionlocal() as session:
        query = select(Document.id).where(Document.company == company)
        if filename:
            query = query.where(Document.filename == filename)
        ids = list(session.scalars(query))
    for document_id in ids:
        with sessionlocal() as session:
            document = session.get(Document, document_id, with_for_update=True)
            if Path(document.filename).suffix.lower() not in {".docx", ".md"}:
                continue
            if document.status not in {"completed", "failed"}:
                continue
            result = {"id": document.id, "filename": document.filename, "status": document.status}
            results.append(result)
            if not apply:
                continue
            document.status = "pending"
            session.commit()
        try:
            task = process_document.apply_async(args=(document_id,), retry=False)
            result.update(status="pending", task_id=task.id)
        except Exception:
            with sessionlocal() as session:
                document = session.get(Document, document_id, with_for_update=True)
                if document.status == "pending":
                    document.status = "failed"
                    session.commit()
            raise
    return results


def main() -> None:
    import json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--company", required=True)
    parser.add_argument("--filename")
    parser.add_argument("--apply", action="store_true", help="Default is a read-only preview")
    args = parser.parse_args()
    print(
        json.dumps(reindex(args.company, args.filename, args.apply), indent=2, ensure_ascii=False)
    )


if __name__ == "__main__":
    main()
