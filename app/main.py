from fastapi import FastAPI

from app.routers.ask import router as ask_router
from app.routers.documents import router as document_router

app = FastAPI()
app.include_router(document_router)
app.include_router(ask_router)


@app.get("/health")
async def healthcheck():
    return {"status": "ok"}
