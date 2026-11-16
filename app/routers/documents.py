import hashlib
from pathlib import Path
from uuid import uuid4

import aiofiles
import aiofiles.os
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_session
from app.models import Document
from app.tasks import process_document

router = APIRouter(prefix="/documents", tags=["file_process"])

FORMATS = {".pdf", ".docx", ".md"}


@router.post("", status_code=202)
async def load_file(
    company: str = Form(...),
    file: UploadFile = File(...),
    session: AsyncSession = Depends(get_session),
):
    if file.filename is None:
        raise HTTPException(detail="Файл не внесен", status_code=422)

    extension = Path(file.filename).suffix.lower()
    if extension not in FORMATS:
        raise HTTPException(detail="Неподходящий формат файла", status_code=415)
    if len(company.strip()) < 1:
        raise HTTPException(detail="Внесите название компании", status_code=422)

    hasher = hashlib.sha256()
    filename = f"{uuid4()}{extension}"
    path = Path("/app/uploads") / filename

    async with aiofiles.open(path, "wb") as output:
        while True:
            data = await file.read(1024 * 1024)

            if not data:
                break

            await output.write(data)
            hasher.update(data)

    content_hash = hasher.hexdigest()
    stmt = select(Document).where(
        Document.company == company.strip(), Document.content_hash == content_hash
    )

    result = await session.execute(stmt)
    existing_document = result.scalar_one_or_none()

    if existing_document is not None:
        await aiofiles.os.remove(path)
        return {
            "id": existing_document.id,
            "status": existing_document.status,
            "duplicate": True,
        }

    document = Document(
        filename=file.filename,
        company=company.strip(),
        file_path=str(path),
        content_hash=content_hash,
        status="pending",
    )
    session.add(document)
    await session.commit()
    process_document.delay(document.id)

    return {"id": document.id, "status": document.status, "duplicate": False}


@router.get("/{document_id}")
async def get_document_by_id(document_id: int, session: AsyncSession = Depends(get_session)):
    document = await session.get(Document, document_id)

    if document is None:
        raise HTTPException(detail="Документ не найден", status_code=404)

    return {
        "id": document.id,
        "filename": document.filename,
        "company": document.company,
        "status": document.status,
    }
