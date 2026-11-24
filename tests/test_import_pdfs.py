from pathlib import Path
from unittest.mock import Mock

import pytest

from scripts.import_pdfs import EXPECTED_COUNTS, find_pdfs, upload_all


def test_discover_pdfs_by_company(tmp_path):
    for company, expected in EXPECTED_COUNTS.items():
        pdf_dir = tmp_path / company / "pdf"
        pdf_dir.mkdir(parents=True)
        for index in range(expected):
            (pdf_dir / f"{index:03}.pdf").write_bytes(b"%PDF-1.4\n")
    discovered = find_pdfs(tmp_path)
    assert len(discovered) == 236
    assert len(set(discovered)) == 236


def test_missing_company_folder_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="Missing PDF directory"):
        find_pdfs(tmp_path)


def test_local_import_uploads_documents_with_correct_company(tmp_path):
    path = tmp_path / "sample.pdf"
    path.write_bytes(b"%PDF-1.4\n")
    client = Mock()
    client.post.return_value.status_code = 202
    client.post.return_value.json.return_value = {
        "id": 42,
        "status": "pending",
        "duplicate": False,
    }
    results, elapsed = upload_all(
        client, [("ZX Bank", Path(path))], "test-secret-with-32-characters-minimum"
    )
    assert results[0]["company"] == "ZX Bank"
    assert results[0]["id"] == 42
    assert elapsed >= 0
    assert client.post.call_args.kwargs["data"]["company"] == "ZX Bank"
