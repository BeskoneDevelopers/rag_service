def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 150):
    if chunk_size <= 0:
        raise ValueError("Размер чанка должен быть больше нуля")
    if chunk_size <= overlap or overlap < 0:
        raise ValueError("Некорректный overlap")

    block_chunk = []
    start_point = 0
    text_length = len(text)

    while start_point < len(text):
        end_point = start_point + chunk_size
        part = text[start_point:end_point].strip()

        if part:
            block_chunk.append(part)
        if end_point >= text_length:
            break

        start_point += chunk_size - overlap

    return block_chunk


def chunk_blocks(blocks: list[dict]) -> list[dict]:
    result_block = []

    for part in blocks:
        chunk = chunk_text(part["text"])
        for part_chunk in chunk:
            block_chunk = {
                "text": part_chunk,
                "page_number": part["page_number"],
                "section": part["section"],
            }
            result_block.append(block_chunk)

    return result_block
