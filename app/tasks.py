from celery.utils.log import get_task_logger

from app.celery_app import celery_app
from app.chunker import chunk_blocks
from app.embeddings import embed_texts
from app.models import Chunk, Document
from app.parser import parser_document
from app.worker_db import sessionlocal

logger = get_task_logger(__name__)


@celery_app.task()
def process_document(document_id: int):

    with sessionlocal() as session:
        document = session.get(Document, document_id)

        if document is None:
            raise ValueError(f"Документ {document_id} не найден")

        try:
            filepath = document.file_path
            document.status = "processing"
            session.commit()
            blocks = parser_document(filepath)
            chunking = chunk_blocks(blocks)

            list_text = [chunk["text"] for chunk in chunking]
            if not list_text:
                raise ValueError("Ошибка подготовки чанков")
            vectors = embed_texts(list_text)

            if len(chunking) != len(vectors):
                raise ValueError("Количество чанков не совпадает с векторами")

            for index, (chunk, vector) in enumerate(zip(chunking, vectors)):
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
            session.rollback()
            document.status = "failed"
            session.commit()
            raise

    return {"processed": document_id}
