from celery import Celery

from app.config import settings

celery_app = Celery(
    "celery_app",
    broker=settings.celery_broker_url,
    include=["app.tasks"],
    backend=settings.celery_result_backend,
    result_expires=3600,
)

celery_app.conf.update(
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    worker_prefetch_multiplier=1,
    task_time_limit=settings.celery_task_time_limit,
    broker_transport_options={"visibility_timeout": settings.celery_visibility_timeout},
    result_backend_transport_options={"visibility_timeout": settings.celery_visibility_timeout},
    visibility_timeout=settings.celery_visibility_timeout,
)
