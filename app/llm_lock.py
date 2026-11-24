import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress
from threading import Lock
from uuid import uuid4

from anyio import CancelScope, current_time, fail_after, sleep
from anyio.lowlevel import checkpoint
from redis.asyncio import Redis
from redis.asyncio.retry import Retry
from redis.backoff import NoBackoff
from redis.exceptions import RedisError

from app.config import settings
from app.llm_client import LLMTimeoutError

logger = logging.getLogger(__name__)

LOCK_KEY = "rag:llm:generation"
LOCK_TTL = 180
REDIS_TIMEOUT = 3
_generation_slot = Lock()


class LLMLockUnavailableError(Exception):
    pass


class LLMBusyError(Exception):
    pass


RELEASE_SCRIPT = """
if redis.call("GET", KEYS[1]) == ARGV[1] then
    return redis.call("DEL", KEYS[1])
end
return 0
"""

RENEW_SCRIPT = """
if redis.call("GET", KEYS[1]) == ARGV[1] then
    return redis.call("EXPIRE", KEYS[1], ARGV[2])
end
return 0
"""


@asynccontextmanager
async def _lock_redis() -> AsyncIterator[Redis]:
    try:
        with fail_after(REDIS_TIMEOUT):
            # A retried SET NX can report contention after the first attempt succeeded.
            async with Redis.from_url(
                settings.celery_broker_url, retry=Retry(NoBackoff(), 0)
            ) as redis:
                yield redis
    except (RedisError, TimeoutError, ValueError) as exc:
        raise LLMLockUnavailableError("Сервис блокировок недоступен") from exc


async def release_llm_lock(token: str) -> None:
    try:
        async with _lock_redis() as redis:
            await redis.eval(RELEASE_SCRIPT, 1, LOCK_KEY, token)
    except LLMLockUnavailableError:
        logger.warning("Failed to release the LLM lock; Redis will expire it", exc_info=True)


async def renew_llm_lock(token: str) -> bool:
    async with _lock_redis() as redis:
        return bool(await redis.eval(RENEW_SCRIPT, 1, LOCK_KEY, token, LOCK_TTL))


async def _keep_llm_lock(token: str, generation: CancelScope) -> None:
    while True:
        await sleep(LOCK_TTL / 3)
        started = current_time()
        if started >= generation.deadline:
            return

        try:
            renewed = await renew_llm_lock(token)
        except LLMLockUnavailableError:
            logger.warning("Failed to renew the LLM lock", exc_info=True)
            generation.cancel()
            return

        if not renewed or current_time() >= generation.deadline:
            generation.cancel()
            return

        generation.deadline = started + LOCK_TTL * 2 / 3


@asynccontextmanager
async def llm_generation_lock() -> AsyncIterator[None]:
    await checkpoint()
    token = uuid4().hex
    acquired = None
    # Hold the local slot through cleanup even if Redis drops the lease unexpectedly.
    if not _generation_slot.acquire(blocking=False):
        raise LLMBusyError("Модель уже выполняет другой запрос")

    try:
        # A disconnect during SET must wait for its bounded result so cleanup can run.
        with CancelScope(shield=True):
            started = current_time()
            async with _lock_redis() as redis:
                acquired = bool(await redis.set(LOCK_KEY, token, nx=True, ex=LOCK_TTL))
        if not acquired:
            raise LLMBusyError("Модель уже выполняет другой запрос")

        # The absolute deadline is independent of the renewable Redis lease.
        try:
            with fail_after(settings.llm_generation_timeout_seconds):
                with CancelScope(deadline=started + LOCK_TTL * 2 / 3) as generation:
                    heartbeat = asyncio.create_task(_keep_llm_lock(token, generation))
                    try:
                        await checkpoint()
                        yield
                    finally:
                        heartbeat.cancel()
                        with CancelScope(shield=True), suppress(asyncio.CancelledError):
                            await heartbeat
        except TimeoutError as exc:
            raise LLMTimeoutError("Превышен абсолютный лимит генерации") from exc

        if generation.cancel_called:
            raise LLMLockUnavailableError("Блокировка генерации потеряна")
    finally:
        try:
            if acquired is not False:
                # SET may have reached Redis even if we couldn't read its reply.
                with CancelScope(shield=True):
                    await release_llm_lock(token)
        finally:
            _generation_slot.release()
