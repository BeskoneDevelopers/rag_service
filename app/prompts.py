import re

from app.models import Chunk


def has_invalid_citation(answer: str, source_count: int) -> bool:
    allowed = {str(number) for number in range(1, source_count + 1)}
    return any(number not in allowed for number in re.findall(r"\[([0-9]+)\]", answer))


def build_context(chunks: list[Chunk]) -> str:
    context_block = []

    for number, chunk in enumerate(chunks, start=1):
        text = (
            f"Источник [{number}]\n"
            f"Документ ID: {chunk.document_id}\n"
            f"Страница: {chunk.page_number}\n"
            f"Раздел: {chunk.section}\n"
            f"Текст: {chunk.content}"
        )
        context_block.append(text)

    return "\n\n".join(context_block)


def build_prompt(question: str, context: str) -> str:
    prompt = f"""Ты помощник, отвечающий на вопросы по документам компании.

Правила:
1. Используй только информацию из предоставленного контекста.
2. Не придумывай отсутствующие сведения.
3. Если информации недостаточно, прямо сообщи об этом.
4. После утверждений указывай источники в формате [1], [2].
5. Используй только номера источников, присутствующие в контексте.
6. Не используй ID документов вместо номеров источников.
7. Если утверждение подтверждается несколькими источниками, укажи их вместе: [1][2].

КОНТЕКСТ:
{context}

ВОПРОС:
{question}

ОТВЕТ:"""

    return prompt
