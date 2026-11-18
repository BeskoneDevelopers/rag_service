import asyncio
import logging

from celery.exceptions import TimeoutError as CeleryTimeoutError
from kombu.exceptions import OperationalError
from redis.exceptions import RedisError

from app.celery_app import celery_app
from app.llm_client import LLMServiceError, generate_answer
from app.llm_lock import llm_generation_lock
from app.prompts import build_prompt, has_invalid_citation

logger = logging.getLogger(__name__)


class RetrievalError(Exception):
    def __init__(self, status_code: int, detail: str):
        super().__init__(detail)
        self.status_code = status_code


async def retrieve_context(company: str, question: str, top_k: int = 5) -> dict:
    try:
        task = await asyncio.to_thread(
            celery_app.send_task,
            "app.tasks.retrieve_context_task",
            args=[company, question, top_k],
            retry=False,
        )
        return await asyncio.to_thread(task.get, timeout=120)
    except CeleryTimeoutError as exc:
        raise RetrievalError(504, "Истекло время поиска документов") from exc
    except Exception as exc:
        # Celery can wrap Redis result-backend connection failures in RuntimeError.
        if isinstance(exc, (OperationalError, RedisError)) or isinstance(exc.__cause__, RedisError):
            raise RetrievalError(503, "Сервис поиска временно недоступен") from exc
        logger.exception("Retrieval task failed")
        raise RetrievalError(502, "Не удалось выполнить поиск по документам") from exc


async def ask_rag(company: str, question: str, top_k: int = 5) -> dict:
    retrieve = await retrieve_context(company, question, top_k)

    if not retrieve["context"]:
        return {"answer": "Не знаю: в документах этого нет", "sources": []}
    prompt = build_prompt(question, retrieve["context"])

    async with llm_generation_lock():
        result = await generate_answer(prompt)
    if has_invalid_citation(result, len(retrieve["sources"])):
        raise LLMServiceError("Ответ llama-server содержит ссылку на неизвестный источник")
    return {"answer": result, "sources": retrieve["sources"]}
