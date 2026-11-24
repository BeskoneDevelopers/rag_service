import pytest
from docx import Document as DocxDocument

from app.chunker import chunk_blocks, chunk_text
from app.parser import parser_document


@pytest.mark.parametrize("extension", ["md", "docx"])
def test_small_paragraphs_are_merged_within_each_section(tmp_path, extension):
    path = tmp_path / f"awards.{extension}"
    if extension == "md":
        path.write_text(
            "# Awards\n\nFirst award.\n\nSecond award.\n\n# Courses\n\nAdmissions.",
            encoding="utf-8",
        )
    else:
        doc = DocxDocument()
        doc.add_heading("Awards", level=1)
        doc.add_paragraph("First award.")
        doc.add_paragraph("Second award.")
        doc.add_heading("Courses", level=2)
        doc.add_paragraph("Admissions.")
        doc.save(path)
    assert chunk_blocks(parser_document(str(path))) == [
        {
            "text": "Awards\n\nFirst award.\n\nSecond award.",
            "page_number": None,
            "section": "Awards",
        },
        {"text": "Courses\n\nAdmissions.", "page_number": None, "section": "Courses"},
    ]


@pytest.mark.parametrize("heading", ["Awards", "Long heading " * 200])
def test_heading_is_in_every_chunk_without_exceeding_size(heading):
    text = "0123456789" * 400
    chunks = chunk_blocks([{"text": text, "page_number": None, "section": heading}])
    prefix = heading[:200] + "\n\n"
    assert all(len(c["text"]) <= 1000 and c["text"].startswith(prefix) for c in chunks)
    assert all(c["section"] == heading for c in chunks)
    bodies = [c["text"][len(prefix) :] for c in chunks]
    assert all(left[-150:] == right[:150] for left, right in zip(bodies, bodies[1:]))
    assert bodies[0] + "".join(body[150:] for body in bodies[1:]) == text


def test_pdf_pages_keep_previous_chunk_boundaries():
    text = "Text " * 400
    blocks = [{"text": text, "page_number": n, "section": None} for n in (1, 2)]
    chunks = chunk_blocks(blocks)
    assert [c["text"] for c in chunks if c["page_number"] == 1] == chunk_text(text)
    assert [c["text"] for c in chunks if c["page_number"] == 2] == chunk_text(text)


@pytest.mark.parametrize("size,overlap", [(0, 0), (100, 100), (100, -1)])
def test_invalid_chunk_limits_are_rejected(size, overlap):
    with pytest.raises(ValueError):
        chunk_text("Fact", size, overlap)
