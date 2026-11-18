from celery.utils.log import get_task_logger
from sqlalchemy.exc import SQLAlchemyError

from app.celery_app import celery_app
from app.chunker import chunk_blocks
from app.embeddings import embed_texts
from app.models import Chunk, Document
from app.parser import parser_document
from app.prompts import build_context
from app.retrieval import SIMILARITY_THRESHOLD, retrieve_chunks
from app.worker_db import sessionlocal

logger = get_task_logger(__name__)


@celery_app.task(acks_late=True, reject_on_worker_lost=True)
def process_document(document_id: int):
    with sessionlocal() as session:
        document = session.get(Document, document_id, with_for_update=True)

        if document is None:
            raise ValueError(f"Документ {document_id} не найден")
        if document.status == "completed":
            return {"processed": document_id}

        # Expose "processing" to GET /documents/{id} before expensive indexing.
        # A redelivered task must also be allowed to resume a processing document.
        document.status = "processing"
        session.commit()

        # Lock the row for the whole indexing transaction. After a worker crash
        # PostgreSQL releases it, so Celery can safely redeliver the same task.
        # A duplicate that waited for the first worker sees "completed" and exits.
        document = session.get(
            Document, document_id, with_for_update=True, populate_existing=True
        )
        if document is None:
            raise ValueError(f"Документ {document_id} не найден")
        if document.status in {"completed", "failed"}:
            return {"processed": document_id}

        try:
            filepath = document.file_path
            blocks = parser_document(filepath)
            chunking = chunk_blocks(blocks)

            list_text = [chunk["text"] for chunk in chunking]
            if not list_text:
                raise ValueError("Ошибка подготовки чанков")
            vectors = embed_texts(list_text)

            if len(chunking) != len(vectors):
                raise ValueError("Количество чанков не совпадает с векторами")

            for index, (chunk, vector) in enumerate(zip(chunking, vectors, strict=True)):
                res = Chunk(
                    document_id=document.id,
                    chunk_index=index,
                    content=chunk["text"],
                    page_number=chunk["page_number"],
                    section=chunk["section"],
                    company=document.company,
                    embedding=vector,
                )
                session.add(res)
            document.status = "completed"
            session.commit()

        except Exception:
            try:
                session.rollback()
                document.status = "failed"
                session.commit()
            except SQLAlchemyError:
                logger.exception("Could not record processing failure for document %s", document_id)
            raise

    return {"processed": document_id}


@celery_app.task
def retrieve_context_task(company: str, question: str, top_k: int = 5) -> dict:
    with sessionlocal() as session:
        chunks = retrieve_chunks(session, company, question, top_k)
        chunks = [
            (chunk, similarity)
            for chunk, similarity in chunks
            if similarity >= SIMILARITY_THRESHOLD
        ]

        if not chunks:
            return {"context": "", "sources": []}

        sources = []

        for number, (chunk, similarity) in enumerate(chunks, start=1):
            sources.append(
                {
                    "number": number,
                    "chunk_id": chunk.id,
                    "document_id": chunk.document_id,
                    "filename": chunk.document.filename,
                    "page_number": chunk.page_number,
                    "section": chunk.section,
                    "company": chunk.company,
                    "score": similarity,
                }
            )

        context_chunks = [chunk for chunk, similarity in chunks]
        context = build_context(context_chunks)
        return {"context": context, "sources": sources}
