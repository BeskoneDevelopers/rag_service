from celery.utils.log import get_task_logger

from app.celery_app import celery_app
from app.chunker import chunk_blocks
from app.models import Document
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
            chunk_blocks(blocks)

        except Exception:
            session.rollback()
            document.status = "failed"
            session.commit()
            raise

    return {"processed": document_id}
