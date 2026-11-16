from sqlalchemy import select
from sqlalchemy.orm import Session

from app.embeddings import embed_texts
from app.models import Chunk


def search_chunks(
    session: Session, company: str, query_vector: list[float], top_k: int = 5
) -> list[Chunk]:
    stmt = (
        select(Chunk)
        .where(Chunk.company == company)
        .order_by(Chunk.embedding.cosine_distance(query_vector))
        .limit(top_k)
    )

    chunks = session.scalars(stmt).all()
    return chunks


def retrieve_chunks(session: Session, company: str, question: str, top_k: int = 5) -> list[Chunk]:
    vectors = embed_texts([question])
    query_vector = vectors[0]
    chunks = search_chunks(session, company, query_vector, top_k)
    return chunks
