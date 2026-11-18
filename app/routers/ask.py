import asyncio
import json
import logging
from collections.abc import AsyncGenerator, AsyncIterator
from contextlib import asynccontextmanager, suppress

from anyio import CancelScope
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, Field
from starlette.requests import ClientDisconnect
from starlette.types import Message, Receive, Scope, Send

from app.auth import get_current_user
from app.llm_client import LLMServiceError, LLMTimeoutError, LLMUnavailableError, stream_answer
from app.llm_lock import LLMBusyError, LLMLockUnavailableError, llm_generation_lock
from app.prompts import build_prompt, has_invalid_citation
from app.rag_pipeline import RetrievalError, ask_rag, retrieve_context

router = APIRouter()
logger = logging.getLogger(__name__)


class AskRequest(BaseModel):
    company: str = Field(min_length=1)
    question: str = Field(min_length=1)
    top_k: int = Field(default=5, ge=1, le=10)


def make_sse(event: str, data: dict | list) -> str:
    payload = json.dumps(data, ensure_ascii=False)
    return f"event: {event}\ndata: {payload}\n\n"


async def _listen_for_disconnect(receive: Receive, request_scope: CancelScope) -> None:
    while (await receive())["type"] != "http.disconnect":
        pass
    request_scope.cancel()


@asynccontextmanager
async def _cancel_on_disconnect(receive: Receive) -> AsyncIterator[None]:
    with CancelScope() as request_scope:
        listener = asyncio.create_task(_listen_for_disconnect(receive, request_scope))
        try:
            yield
        finally:
            listener.cancel()
            with CancelScope(shield=True), suppress(asyncio.CancelledError):
                await listener


class LLMStreamingResponse(StreamingResponse):
    def __init__(self, tokens: AsyncGenerator[str, None], sources: list[dict]):
        super().__init__(tokens, media_type="text/event-stream")
        self.tokens = tokens
        self.sources = sources

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        started = False

        async def send_or_disconnect(message: Message) -> None:
            try:
                await send(message)
            except OSError as exc:
                raise ClientDisconnect() from exc

        async with _cancel_on_disconnect(receive):
            answer_parts = []
            try:
                async with llm_generation_lock():
                    try:
                        await send_or_disconnect(
                            {
                                "type": "http.response.start",
                                "status": self.status_code,
                                "headers": self.raw_headers,
                            }
                        )
                        started = True
                        async for token in self.tokens:
                            answer_parts.append(token)
                            await send_or_disconnect(
                                {
                                    "type": "http.response.body",
                                    "body": make_sse("token", {"text": token}).encode(self.charset),
                                    "more_body": True,
                                }
                            )
                    finally:
                        with CancelScope(shield=True):
                            await self.tokens.aclose()
                if has_invalid_citation("".join(answer_parts), len(self.sources)):
                    raise LLMServiceError("Ответ Ollama содержит ссылку на неизвестный источник")
                terminal_body = make_sse("sources", self.sources).encode(self.charset)
            except ClientDisconnect:
                return
            except (
                LLMBusyError,
                LLMLockUnavailableError,
                LLMUnavailableError,
                LLMTimeoutError,
                LLMServiceError,
            ) as exc:
                if not started:
                    raise HTTPException(status_code=503, detail=str(exc)) from exc
                terminal_body = make_sse("error", {"detail": str(exc)}).encode(self.charset)
            except Exception as exc:
                logger.exception("Unexpected error while streaming an LLM answer")
                detail = "Не удалось завершить поток ответа"
                if not started:
                    raise HTTPException(status_code=500, detail=detail) from exc
                terminal_body = make_sse("error", {"detail": detail}).encode(self.charset)

            with suppress(OSError):
                await send(
                    {
                        "type": "http.response.body",
                        "body": terminal_body,
                        "more_body": False,
                    }
                )


@router.post("/ask")
async def ask(
    request: AskRequest, http_request: Request, current_user: dict = Depends(get_current_user)
):
    try:
        if request.company != current_user["company"]:
            raise HTTPException(
                status_code=403,
                detail="Нет доступа к документам компании",
            )

        async with _cancel_on_disconnect(http_request.receive):
            return await ask_rag(current_user["company"], request.question, request.top_k)

    except LLMUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except LLMTimeoutError as exc:
        raise HTTPException(status_code=504, detail=str(exc)) from exc
    except LLMServiceError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except (LLMBusyError, LLMLockUnavailableError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except RetrievalError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc

    return Response(status_code=204)


@router.post("/ask/stream")
async def ask_stream(
    request: AskRequest,
    current_user: dict = Depends(get_current_user),
):
    if request.company != current_user["company"]:
        raise HTTPException(
            status_code=403,
            detail="Нет доступа к документам компании",
        )

    try:
        retrieve = await retrieve_context(current_user["company"], request.question, request.top_k)
    except RetrievalError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc

    if not retrieve["context"]:

        async def no_context_events():
            yield make_sse("token", {"text": "Не знаю: в документах этого нет"})
            yield make_sse("sources", retrieve["sources"])

        return StreamingResponse(no_context_events(), media_type="text/event-stream")

    prompt = build_prompt(request.question, retrieve["context"])
    return LLMStreamingResponse(stream_answer(prompt), retrieve["sources"])
