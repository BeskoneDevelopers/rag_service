from sqlalchemy import select, text
from sqlalchemy.orm import Session, selectinload

from app.embeddings import embed_texts
from app.models import Chunk

SIMILARITY_THRESHOLD = 0.45


def search_chunks(
    session: Session, company: str, query_vector: list[float], top_k: int = 5
) -> list[tuple[Chunk, float]]:
    block_chunk = []
    distance = Chunk.embedding.cosine_distance(query_vector)

    session.execute(text("SET LOCAL hnsw.iterative_scan = strict_order"))

    stmt = (
        select(Chunk, distance.label("distance"))
        .options(selectinload(Chunk.document))
        .where(Chunk.company == company)
        .order_by(distance)
        .limit(top_k)
    )

    chunks = session.execute(stmt).all()

    for chunk, dist in chunks:
        similarity = 1 - dist
        block_chunk.append((chunk, similarity))

    return block_chunk


def retrieve_chunks(
    session: Session, company: str, question: str, top_k: int = 5
) -> list[tuple[Chunk, float]]:
    vectors = embed_texts([question])
    query_vector = vectors[0]
    chunks = search_chunks(session, company, query_vector, top_k)
    return chunks
