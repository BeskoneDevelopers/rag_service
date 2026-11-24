from pathlib import Path

import pymupdf
from docx import Document as DocxDocument
from markdown_it import MarkdownIt


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
    section_name = None
    paragraphs = []

    for paragraph in doc.paragraphs:
        text = paragraph.text.strip()

        if paragraph.style.name in ("Heading 1", "Heading 2"):
            if paragraphs:
                block_text.append(
                    {"text": "\n\n".join(paragraphs), "page_number": None, "section": section_name}
                )
                paragraphs = []
            section_name = text
            continue

        if text:
            paragraphs.append(text)

    if paragraphs:
        block_text.append(
            {"text": "\n\n".join(paragraphs), "page_number": None, "section": section_name}
        )

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
    tokens = MarkdownIt().parse(doc)
    block_text = []
    section_name = None
    paragraphs = []

    for i, token in enumerate(tokens):
        if token.type == "heading_open":
            if paragraphs:
                block_text.append(
                    {"text": "\n\n".join(paragraphs), "page_number": None, "section": section_name}
                )
                paragraphs = []
            section_name = tokens[i + 1].content
            continue

        if token.type == "inline":
            if i > 0 and tokens[i - 1].type == "heading_open":
                continue
            text = token.content.strip()

            if text:
                paragraphs.append(text)
    if paragraphs:
        block_text.append(
            {"text": "\n\n".join(paragraphs), "page_number": None, "section": section_name}
        )
    return block_text
