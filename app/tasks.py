from app.celery_app import celery_app
from celery.utils.log import get_task_logger

logger = get_task_logger(__name__)

@celery_app.task()
def print_message(word: str) -> dict[str, str]:
    logger.info(f"Работаю со словом {word}")
    return {"processed": word}