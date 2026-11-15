from app.celery_app import celery_app
from celery.utils.log import get_task_logger
from app.worker_db import sessionlocal
from app.models import Document
from app.parser import parser_document


logger = get_task_logger(__name__)

@celery_app.task()
def process_document(document_id: int):

    with sessionlocal() as session:
        logger.info(f"Взят документ {document_id}")
        document = session.get(Document, document_id)

        if document is None:
            raise ValueError(f"Документ {document_id} не найден")

        logger.info(f"Найден документ {document.file_path}")

        blocks = parser_document(document.file_path)
        logger.info(f"Колл абзацев {len(blocks)}")

    return {"processed": document_id}