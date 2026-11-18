import pymupdf
import pytest
from docx import Document as DocxDocument

from app.parser import parser_document, parser_docx, parser_md


def test_markdown_sections(tmp_path):
    path = tmp_path / "test.md"
    path.write_text(
        "# Компания\n\nОписание компании.\n\n## Финансы\n\nВыручка выросла.",
        encoding="utf-8",
    )

    blocks = parser_md(str(path))

    assert blocks[0]["section"] == "Компания"
    assert blocks[1]["section"] == "Финансы"


def test_docx_sections(tmp_path):
    path = tmp_path / "test.docx"

    doc = DocxDocument()
    doc.add_paragraph("Финансы", style="Heading 1")
    doc.add_paragraph("Выручка выросла.")
    doc.save(path)

    blocks = parser_docx(str(path))

    assert len(blocks) == 1
    assert blocks[0]["section"] == "Финансы"
    assert blocks[0]["text"] == "Выручка выросла."


def test_pdf_page_numbers_skip_empty_pages(tmp_path):
    path = tmp_path / "test.PDF"
    with pymupdf.open() as document:
        document.new_page().insert_text((72, 72), "First page")
        document.new_page()
        document.new_page().insert_text((72, 72), "Third page")
        document.save(path)

    assert parser_document(str(path)) == [
        {"text": "First page", "page_number": 1, "section": None},
        {"text": "Third page", "page_number": 3, "section": None},
    ]


def test_unknown_extension(tmp_path):
    with pytest.raises(ValueError, match="Неизвестный формат"):
        parser_document(str(tmp_path / "unsupported.txt"))
