import pytest
from pydantic import ValidationError

from app.celery_app import celery_app
from app.config import Settings, settings


def test_worker_acknowledgements_and_visibility_cover_execution_limit():
    assert celery_app.conf.task_acks_late is True
    assert celery_app.conf.task_reject_on_worker_lost is True
    assert celery_app.conf.worker_prefetch_multiplier == 1
    assert celery_app.conf.task_time_limit == settings.celery_task_time_limit
    timeout = settings.celery_visibility_timeout
    assert timeout > settings.celery_task_time_limit
    assert celery_app.conf.broker_transport_options["visibility_timeout"] == timeout
    assert celery_app.conf.result_backend_transport_options["visibility_timeout"] == timeout
    assert celery_app.conf.visibility_timeout == timeout


@pytest.mark.parametrize(
    "overrides",
    [
        {"llm_max_tokens": 0},
        {"llm_generation_timeout_seconds": 0},
        {"celery_task_time_limit": 600, "celery_visibility_timeout": 600},
    ],
)
def test_invalid_limits_fail_at_startup(overrides):
    with pytest.raises(ValidationError):
        Settings(**{**settings.model_dump(), **overrides})
