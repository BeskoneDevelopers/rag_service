# RAG-сервис по документам компании

FastAPI принимает PDF, DOCX и Markdown, индексирует их в фоне и отвечает на вопросы по документам компании из JWT.

## Архитектура

1. `POST /documents` сохраняет файл в общий Docker-том, считает SHA-256 и публикует задачу Celery. PostgreSQL гарантирует уникальность пары `company + content_hash`, в том числе при параллельной загрузке.
2. Celery разбирает документ, делит текст на чанки, считает векторы `BAAI/bge-m3` и записывает их в PostgreSQL. У документа есть статусы `pending`, `processing`, `completed` и `failed`. В API статус можно проверить по ID.
3. `/ask` и `/ask/stream` передают поиск worker. pgvector выбирает ближайшие чанки компании по косинусному расстоянию через HNSW. Чанки ниже порога отсекаются до построения контекста.
4. FastAPI передаёт контекст локальной Ollama с `qwen3:1.7b`. Redis Lock разрешает одну генерацию для обоих маршрутов. Обычный ответ содержит `answer` и `sources`. Потоковый ответ отдаёт `token` и завершается событием `sources` или `error`.

Celery нужен, чтобы загрузка файла не ждала парсинга, расчёта embedding и записи всех чанков. Поиск также выполняется в worker, поэтому API не загружает embedding-модель. Redis хранит очередь, результаты поиска и ключ блокировки генерации.

### Чанки и порог

- Чанк содержит до 1000 символов с перекрытием 150. Перекрытие сохраняет часть текста на границе. Чанки не пересекают исходные блоки парсера, поэтому сохраняют страницу PDF или раздел DOCX и Markdown.
- По умолчанию берутся до пяти ближайших чанков (`top_k=5`). В контекст попадают только результаты с `similarity >= 0.45`, где similarity равен `1 - cosine_distance`. Порог выбран на локальной выборке, описанной ниже.
- PDF обрабатывается по текстовому слою. Для изображения страницы требуется предварительный OCR. DOCX читает абзацы и заголовки уровней 1–2. Markdown использует заголовки для определения разделов.

## Запуск

Понадобятся Docker с Compose v2, GNU Make для команды `make up` и Ollama на хосте с локальной моделью `qwen3:1.7b`. Для тестов вне Docker нужен Python 3.14.

При первом запуске скопируйте `.env.example` в `.env` и заполните переменные. Для Compose адрес базы в URL — `db:5432`. `JWT_SECRET` держите в локальном `.env`. Значение можно сгенерировать командой `python -c "import secrets; print(secrets.token_urlsafe(32))"`.

| Переменная | Назначение |
| --- | --- |
| `POSTGRES_USER`, `POSTGRES_PASSWORD`, `POSTGRES_DB` | Пользователь, пароль и имя базы для контейнера PostgreSQL |
| `DATABASE_URL` | URL API с драйвером `postgresql+asyncpg` |
| `DATABASE_SYNC_URL` | URL worker и Alembic с драйвером `postgresql+psycopg` |
| `DEBUG` | SQL-логирование в worker |
| `JWT_SECRET` | Ключ подписи JWT (HS256) |
| `CELERY_BROKER_URL` | Очередь Celery и Redis Lock. По умолчанию `redis://redis:6379/0` |
| `CELERY_RESULT_BACKEND` | Результаты задач. По умолчанию `redis://redis:6379/1` |
| `OLLAMA_URL` | URL генерации на хосте. Для Docker Desktop `http://host.docker.internal:11434/api/generate` |

`make up` собирает и запускает сервисы, затем применяет миграции Alembic. Эти же команды можно выполнить в PowerShell.

```sh
docker compose up -d --build
docker compose exec -T web alembic upgrade head
curl http://localhost:8000/health
```

Откройте Swagger по адресу [localhost:8000/docs](http://localhost:8000/docs). На Linux для Ollama укажите адрес хоста, доступный из контейнера. При первом расчёте embedding worker использует том `model_cache` для `BAAI/bge-m3`. Для запуска без доступа к сети заранее нужен заполненный кэш.

### Примеры запросов

API проверяет подпись, срок действия и поля `sub` и `company`. Токен выдаётся вне сервиса. Для локального стенда пример ниже создаёт токен в терминале из ключа, уже переданного контейнеру. Команды используют синтаксис Bash.

```sh
TOKEN=$(docker compose exec -T web python -c 'import os,time,jwt; print(jwt.encode({"sub":"demo","company":"demo","exp":int(time.time())+3600},os.environ["JWT_SECRET"],algorithm="HS256"))')

curl -X POST http://localhost:8000/documents \
  -H "Authorization: Bearer $TOKEN" \
  -F 'company=demo' -F 'file=@example.pdf'

curl http://localhost:8000/documents/1 -H "Authorization: Bearer $TOKEN"

curl -X POST http://localhost:8000/ask \
  -H "Authorization: Bearer $TOKEN" -H 'Content-Type: application/json' \
  -d '{"company":"demo","question":"Какие документы нужны?","top_k":3}'

curl -N -X POST http://localhost:8000/ask/stream \
  -H "Authorization: Bearer $TOKEN" -H 'Content-Type: application/json' \
  -d '{"company":"demo","question":"Какие документы нужны?","top_k":3}'
```

В запрос статуса подставьте ID из ответа загрузки. Дождитесь `completed` перед поиском. Повторная загрузка тех же байтов для компании возвращает прежний ID с `duplicate: true`. Для документа в `failed` такая загрузка повторно ставит индексацию в очередь. При недоступном брокере загрузка отвечает HTTP 503.

У каждого элемента `sources` есть `number`, `chunk_id`, `document_id`, имя файла, страница или раздел и `score`. Номер `[1]` в тексте соответствует `sources` с `number: 1`. Сервис проверяет диапазон номеров. HTTP 502 или финальный `error` в SSE сообщает о ссылке на отсутствующий источник. Уже переданные токены остаются у SSE-клиента. Проверка номера не доказывает, что источник подтверждает утверждение. Без подходящих чанков ответ — «Не знаю: в документах этого нет» и пустые `sources`.

## Тесты и CI

```sh
python -m pip install -e '.[dev]'
make lint
make test
```

В PowerShell используйте `python -m ruff check .` и `python -m pytest -q`. Тесты работают без БД, Redis, Celery worker и Ollama. Подмены проверяют запросы к ним. `conftest.py` запрещает загрузку embedding-модели. GitHub Actions выполняет Ruff и pytest для push и pull request.

## Benchmark retrieval

Понадобятся проиндексированные PDF из RAG-Multi-Corpus и CSV вопросов. При создании индекса передавайте имя компании из CSV в JWT и `POST /documents`. Для PDF из группы CloudWay используйте `CloudWay-24`, как в контрольных запросах скрипта. Дождитесь `completed` для загруженных документов. [PDF корпуса](https://github.com/udayallu/RAG-Multi-Corpus/tree/39071af3f4dd25e59f5c59a6f9b6e8e99cd643b3/datasets), [CSV вопросов](https://raw.githubusercontent.com/udayallu/RAG-Multi-Corpus/39071af3f4dd25e59f5c59a6f9b6e8e99cd643b3/bechmark/bechmark-agentic-references/Dataset%20categories%20-%20queries.csv).

CSV содержит колонки `Enterprise Name`, `Query`, `Supporting Facts`. В последней колонке ожидается JSON-массив объектов с полем `filename`. Скрипт печатает SHA-256 входного CSV, параметры выборки, результаты для разных порогов и Hit@5. С тем же CSV, seed и индексом состав выборки повторяется.

Если CSV лежит только в `/tmp` запущенного worker, сначала сохраните его на хост. Например, `docker compose cp worker:/tmp/rag_benchmark.csv /path/to/rag_benchmark.csv`. Временная папка контейнера не входит в именованный том.

```sh
docker compose cp ./rag_benchmark.csv worker:/tmp/rag_benchmark.csv
docker compose exec -T -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 worker \
  python -m scripts.evaluate_retrieval \
  --benchmark /tmp/rag_benchmark.csv --seed 42 --questions-per-company 20
```

Результат повторной проверки с кэшем `BAAI/bge-m3` и существующим индексом — **76/78 (97,4%)** с порогом и без него. Из 80 вопросов у двух не оказалось эталонного PDF в индексе, поэтому знаменатель — 78. Все 25 проверок вне темы отклонены при 0.45. Это результат указанной выборки.

SHA-256 использованного CSV — `58be4e8ebe96147ab63b159b3166357ad905989023956038ed563f9b88c2d24e`. Исходный CSV и документы нужны для повтора этого результата.

### TTFT

```sh
docker compose exec -T web python -m scripts.measure_ttft --warmup --company 'ZX Bank' \
  --question 'What awards did ZX Bank Ltd. win in 2023?' \
  --question 'Why was ZX Bank recognized for its digital initiatives in 2023?' \
  --question 'Why might ZX Bank consider Phuentsholing an important location for establishing a bank branch?'
```

Скрипт проходит `top_k=2/3/5` для каждого вопроса. `ttft_ms` включает поиск через Celery, построение промпта и ожидание первого фрагмента текста. `llm_ttft_ms` начинается перед вызовом Ollama. Оба замера выполняются внутри приложения без HTTP-доставки SSE клиенту. Число входных и выходных токенов берётся из `prompt_eval_count` и `eval_count` Ollama. `--warmup` исключает из отчёта один предварительный запрос. См. [результаты прогона](benchmarks/ttft_ollama.csv) и [формат ответа Ollama](https://docs.ollama.com/api/generate).

Один прогон после прогрева на `qwen3:1.7b`, по три вопроса на значение `top_k`.

| `top_k` | Медиана `ttft_ms` | Медиана `llm_ttft_ms` | Входные токены | Выходные токены |
| ---: | ---: | ---: | ---: | ---: |
| 2 | 832,5 | 218,8 | 516–989 | 91–98 |
| 3 | 226,5 | 119,7 | 777–1376 | 77–92 |
| 5 | 383,2 | 223,3 | 1379–2057 | 81–107 |

Время PostgreSQL и Celery менялось между запросами. По трём замерам нельзя выбрать самый быстрый `top_k` для всего корпуса.

## Переход на llama.cpp для задания ментора

Текущий запуск использует Ollama и `sentence-transformers`. Для сдачи по требованиям ментора нужны два локальных процесса `llama-server` на хосте и файлы `Qwen3.5-2B-GGUF Q4_K_M` и `bge-m3-GGUF Q4_K_M` из ModelScope.

1. В списках файлов ModelScope подтверждены `Qwen3.5-2B-Q4_K_M.gguf` и `bge-m3-Q4_K_M.gguf`. Перед запуском зафиксировать ревизию и SHA-256 фактически полученных файлов. [Qwen на ModelScope](https://www.modelscope.cn/models/unsloth/Qwen3.5-2B-GGUF), [bge-m3 на ModelScope](https://www.modelscope.cn/models/gpustack/bge-m3-GGUF).
2. После получения файлов запустить Qwen с `llama-server -m <QWEN.gguf> --host <host-ip> --port 8081 --parallel 1`, а bge-m3 с `llama-server -m <BGE.gguf> --host <host-ip> --port 8082 --embeddings --pooling cls`. Подставить адрес хоста, доступный из сети контейнеров. Подключить `/v1/chat/completions` в `llm_client.py` и `/v1/embeddings` в `embeddings.py`. Публичные маршруты FastAPI и Redis Lock можно сохранить. [Документация llama-server](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md).
3. Проверить порядок результатов batch, pooling и размерность 1024. Построить отдельный индекс на новом backend для документов и запросов. Затем повторить Hit@5, проверку порога, ссылки и TTFT. После сравнения можно переключить сервис на новый индекс, сохранив исходный для отката.

Краткий разбор для подготовки к проверке находится в [docs/razbor.md](docs/razbor.md).
