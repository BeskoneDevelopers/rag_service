from fastapi import FastAPI

from app.routers.documents import router as document_router

app = FastAPI()
app.include_router(document_router)


@app.get("/health")
async def healthcheck():
    return {"status": "ok"}
