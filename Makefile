.PHONY: up lint test

up:
	docker compose up -d --build
	docker compose exec -T web alembic upgrade head

lint:
	python -m ruff check .

test:
	python -m pytest -q
