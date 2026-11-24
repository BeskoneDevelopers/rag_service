import asyncio
import hashlib
import logging
from contextlib import suppress
from pathlib import Path
from uuid import uuid4

import aiofiles
import aiofiles.os
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from kombu.exceptions import OperationalError
from redis.exceptions import RedisError
from sqlalchemy import select, update
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth import get_current_user
from app.database import get_session
from app.models import Document
from app.tasks import process_document

router = APIRouter(prefix="/documents", tags=["file_process"])
logger = logging.getLogger(__name__)

FORMATS = {".pdf", ".docx", ".md"}
UPLOAD_DIR = Path("/app/uploads")


@router.post("", status_code=202)
async def load_file(
    company: str = Form(...),
    file: UploadFile = File(...),
    session: AsyncSession = Depends(get_session),
    current_user: dict = Depends(get_current_user),
):
    authorized_company = current_user["company"]

    if company.strip() != authorized_company:
        raise HTTPException(
            status_code=403,
            detail="Нет доступа к документам компании",
        )
    if file.filename is None:
        raise HTTPException(detail="Файл не внесен", status_code=422)

    extension = Path(file.filename).suffix.lower()

    if extension not in FORMATS:
        raise HTTPException(detail="Неподходящий формат файла", status_code=415)

    hasher = hashlib.sha256()
    path = UPLOAD_DIR / f"{uuid4()}{extension}"
    keep_upload = False

    try:
        async with aiofiles.open(path, "wb") as output:
            while data := await file.read(1024 * 1024):
                await output.write(data)
                hasher.update(data)

        content_hash = hasher.hexdigest()
        stmt = (
            insert(Document)
            .values(
                filename=file.filename,
                company=authorized_company,
                file_path=str(path),
                content_hash=content_hash,
                status="pending",
            )
            .on_conflict_do_nothing(constraint="uq_document_company_hash")
            .returning(Document.id)
        )
        document_id = await session.scalar(stmt)
        if document_id is not None:
            # Keep the file once commit starts: an interrupted commit may still succeed.
            keep_upload = True
            await session.commit()
            duplicate = False
        else:
            document = (
                await session.execute(
                    select(Document.id, Document.status).where(
                        Document.company == authorized_company,
                        Document.content_hash == content_hash,
                    )
                )
            ).one()
            document_id = document.id
            duplicate = True
            if document.status != "failed":
                return {"id": document_id, "status": document.status, "duplicate": True}

            claimed = await session.scalar(
                update(Document)
                .where(Document.id == document_id, Document.status == "failed")
                .values(status="pending")
                .returning(Document.id)
            )
            if claimed is None:
                status = await session.scalar(
                    select(Document.status).where(Document.id == document_id)
                )
                return {"id": document_id, "status": status, "duplicate": True}
            await session.commit()
    finally:
        if not keep_upload:
            with suppress(FileNotFoundError):
                await aiofiles.os.remove(path)

    try:
        await asyncio.to_thread(process_document.apply_async, args=(document_id,), retry=False)
    except (OperationalError, RedisError, RuntimeError) as exc:
        logger.warning("Could not enqueue document %s (%s)", document_id, type(exc).__name__)
        try:
            await session.execute(
                update(Document)
                .where(Document.id == document_id, Document.status == "pending")
                .values(status="failed")
            )
            await session.commit()
        except SQLAlchemyError:
            logger.exception("Could not record enqueue failure for document %s", document_id)
        raise HTTPException(
            status_code=503, detail="Очередь индексации временно недоступна"
        ) from exc

    return {"id": document_id, "status": "pending", "duplicate": duplicate}


@router.get("/{document_id}")
async def get_document_by_id(
    document_id: int,
    session: AsyncSession = Depends(get_session),
    current_user: dict = Depends(get_current_user),
):
    stmt = select(Document).where(
        Document.id == document_id, Document.company == current_user["company"]
    )
    document = await session.scalar(stmt)

    if document is None:
        raise HTTPException(detail="Документ не найден", status_code=404)

    return {
        "id": document.id,
        "filename": document.filename,
        "company": document.company,
        "status": document.status,
    }
