from pathlib import Path

import pymupdf
from docx import Document as DocxDocument


def parser_document(file_path: str):
    extension = Path(file_path).suffix.lower()
    parsers = {".docx": parser_docx, ".pdf": parser_pdf, ".md": parser_md}
    action = parsers.get(extension)

    if action is None:
        raise ValueError("Неизвестный формат документа")

    return action(file_path)


def parser_docx(file_path: str):
    doc = DocxDocument(file_path)
    block_text = []
    for paragraph in doc.paragraphs:
        text = paragraph.text.strip()
        block = {"text": text, "page_number": None, "section": None}
        if text:
            block_text.append(block)
    return block_text


def parser_pdf(file_path: str):
    block_text = []
    with pymupdf.open(file_path) as doc:
        for page in doc:
            text = page.get_text().strip()
            page_number = page.number + 1
            if text:
                block = {"text": text, "page_number": page_number, "section": None}
                block_text.append(block)

    return block_text


def parser_md(file_path: str):
    doc = Path(file_path).read_text(encoding="utf-8")
    lines = doc.splitlines()
    block_text = []

    for line in lines:
        text = line.strip()
        block = {"text": text, "page_number": None, "section": None}
        if text:
            block_text.append(block)
    return block_text
