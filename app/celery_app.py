import os

from celery import Celery

celery_app = Celery(
    "celery_app",
    broker=os.getenv("CELERY_BROKER_URL"),
    include=["app.tasks"],
    backend=os.getenv("CELERY_RESULT_BACKEND"),
    result_expires=3600,
)
