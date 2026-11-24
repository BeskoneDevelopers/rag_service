# RAG-сервис по документам компании

FastAPI принимает PDF, DOCX и Markdown, индексирует их в фоне и отвечает на вопросы по документам компании из JWT.

## Архитектура

1. `POST /documents` сохраняет файл в общий Docker-том, считает SHA-256 и публикует задачу Celery. PostgreSQL гарантирует уникальность пары `company + content_hash`, в том числе при параллельной загрузке.
2. Celery разбирает документ, делит текст на чанки, получает векторы от отдельного `llama-server` с `bge-m3 Q4_K_M` и записывает их в PostgreSQL. У документа есть статусы `pending`, `processing`, `completed` и `failed`. В API статус можно проверить по ID.
3. `/ask` и `/ask/stream` передают поиск worker. pgvector выбирает ближайшие чанки компании по косинусному расстоянию через HNSW. Чанки ниже порога отсекаются до построения контекста.
4. FastAPI передаёт контекст локальному `llama-server` с `Qwen3.5-2B Q4_K_M`. Redis Lock разрешает одну генерацию для обоих маршрутов. Обычный ответ содержит `answer` и `sources`. Потоковый ответ отдаёт `token` и завершается событием `sources` или `error`.

Celery нужен, чтобы загрузка файла не ждала парсинга, расчёта embedding и записи всех чанков. Поиск также выполняется в worker, поэтому API не загружает embedding-модель. Redis хранит очередь, результаты поиска и ключ блокировки генерации.

### Чанки и порог

- Чанк содержит до 1000 символов с перекрытием 150. PDF сохраняет прежние границы страниц. Небольшие абзацы DOCX и Markdown объединяются внутри раздела; каждый чанк включает его заголовок в текст для embedding. Заголовок занимает часть лимита 1000 символов (в тексте максимум 200 символов, в метаданных он полный). Разделы не смешиваются. После обновления нужно переиндексировать DOCX/Markdown — [команды](docs/verification.md).
- По умолчанию берутся до пяти ближайших чанков (`top_k=5`). В контекст попадают только результаты с `similarity >= 0.45`, где similarity равен `1 - cosine_distance`. Порог подтверждён проверкой ментора, описанной ниже.
- PDF обрабатывается по текстовому слою. Для изображения страницы требуется предварительный OCR. DOCX читает абзацы и заголовки уровней 1–2. Markdown использует заголовки для определения разделов.

## Запуск

Понадобятся Docker с Compose v2, GNU Make для команды `make up`, Python 3.14 для локальных тестов и два запущенных `llama-server` вне Docker.

При первом запуске скопируйте `.env.example` в `.env` и заполните переменные. Для Compose адрес базы в URL — `db:5432`. `JWT_SECRET` держите в локальном `.env`. Значение можно сгенерировать командой `python -c "import secrets; print(secrets.token_urlsafe(32))"`.

| Переменная | Назначение |
| --- | --- |
| `POSTGRES_USER`, `POSTGRES_PASSWORD`, `POSTGRES_DB` | Пользователь, пароль и имя базы для контейнера PostgreSQL |
| `DATABASE_URL` | URL API с драйвером `postgresql+asyncpg` |
| `DATABASE_SYNC_URL` | URL worker и Alembic с драйвером `postgresql+psycopg` |
| `DEBUG` | SQL-логирование API и worker, по умолчанию `false` |
| `JWT_SECRET` | Ключ подписи JWT (HS256) |
| `CELERY_BROKER_URL` | Очередь Celery и Redis Lock. По умолчанию `redis://redis:6379/0` |
| `CELERY_RESULT_BACKEND` | Результаты задач. По умолчанию `redis://redis:6379/1` |
| `CELERY_TASK_TIME_LIMIT`, `CELERY_VISIBILITY_TIMEOUT` | Максимум задачи 600 с; повторная доставка неподтверждённой задачи после 900 с и очередной проверки Redis. Второе значение должно быть больше первого |
| `LLM_CHAT_URL`, `LLM_MODEL`, `LLM_TIMEOUT_SECONDS` | URL генерации через `llama-server`, имя модели и таймаут |
| `LLM_MAX_TOKENS`, `LLM_GENERATION_TIMEOUT_SECONDS` | 512 токенов и абсолютный лимит 60 с для обоих маршрутов; повтор подключения входит в общий лимит |
| `EMBEDDINGS_URL`, `EMBEDDINGS_MODEL`, `EMBEDDINGS_BATCH_SIZE`, `EMBEDDINGS_TIMEOUT_SECONDS` | URL embedding-сервера, имя модели, размер batch и таймаут |

`make up` собирает и запускает сервисы, затем применяет миграции Alembic. Эти же команды можно выполнить в PowerShell.

```sh
docker compose up -d --build
docker compose exec -T web alembic upgrade head
curl http://localhost:8000/health
```

Откройте Swagger по адресу [localhost:8000/docs](http://localhost:8000/docs). На Linux для llama-server укажите адрес хоста, доступный из контейнера. Embedding-модель работает вне Docker по HTTP. Кэш `sentence-transformers` больше не используется.

### Запросы и запись ответов

API проверяет подпись, срок действия и поля `sub` и `company`. Токен выдаётся вне сервиса. Для локального стенда команды ниже создают его из ключа контейнера. Синтаксис Bash; файл должен существовать в указанном пути.

```sh
TOKEN=$(docker compose exec -T web python -c 'import os,time,jwt; print(jwt.encode({"sub":"demo","company":"demo","exp":int(time.time())+3600},os.environ["JWT_SECRET"],algorithm="HS256"))')

curl -X POST http://localhost:8000/documents \
  -H "Authorization: Bearer $TOKEN" \
  -F 'company=demo' -F 'file=@Awards & Recognitions.docx'

curl http://localhost:8000/documents/1 -H "Authorization: Bearer $TOKEN"

curl -X POST http://localhost:8000/ask \
  -H "Authorization: Bearer $TOKEN" -H 'Content-Type: application/json' \
  -d '{"company":"demo","question":"Какие документы нужны?","top_k":3}'

curl -N -X POST http://localhost:8000/ask/stream \
  -H "Authorization: Bearer $TOKEN" -H 'Content-Type: application/json' \
  -d '{"company":"demo","question":"Какие документы нужны?","top_k":3}'
```

В запрос статуса подставьте ID из ответа загрузки. Дождитесь `completed` перед поиском. Повторная загрузка тех же байтов для компании возвращает прежний ID с `duplicate: true`. Для документа в `failed` такая загрузка повторно ставит индексацию в очередь. При недоступном брокере загрузка отвечает HTTP 503.

Скрипт [capture_responses.py](scripts/capture_responses.py) сохраняет фактические байты `/ask` и SSE с финальным `sources`, не записывая JWT. Команды — в [протоколе проверки](docs/verification.md). Реальные ответы на этом компьютере не записаны: модели и корпус отсутствуют. Тестовые подмены не являются артефактами работы моделей.

У каждого элемента `sources` есть `number`, `chunk_id`, `document_id`, имя файла, страница или раздел и `score`. Номер `[1]` в тексте соответствует `sources` с `number: 1`. Сервис проверяет диапазон номеров. HTTP 502 или финальный `error` в SSE сообщает о ссылке на отсутствующий источник. Уже переданные токены остаются у SSE-клиента. Проверка номера не доказывает, что источник подтверждает утверждение. Без подходящих чанков ответ — «Не знаю: в документах этого нет» и пустые `sources`.

Абсолютный таймаут генерации возвращает 504 до начала SSE, затем — событие `error`. Достижение `max_tokens` считается незавершённым ответом (502 или `error`), а не успешным `sources`. HTTP-поток закрывается до освобождения Redis lock. При недоступном Redis освобождение выполняется через TTL; немедленное удаление ключа при сетевом сбое гарантировать нельзя. Недоступность embedding-сервиса при поиске возвращает 503.

## Тесты и CI

```sh
python -m pip install -e '.[dev]'
make lint
make test
```

В PowerShell используйте `python -m ruff check .` и `python -m pytest -q`. Тесты работают без БД, Redis, Celery worker и llama-server. Подмены проверяют запросы к ним. `conftest.py` запрещает загрузку embedding-модели. GitHub Actions выполняет Ruff и pytest для push и pull request.

## Запуск llama.cpp на компьютере

Скачайте из ModelScope [Qwen3.5-2B Q4_K_M](https://www.modelscope.cn/models/unsloth/Qwen3.5-2B-GGUF) и [bge-m3 Q4_K_M](https://www.modelscope.cn/models/gpustack/bge-m3-GGUF). Установите `llama-server` из [llama.cpp](https://github.com/ggml-org/llama.cpp/releases). Запускайте процессы в двух разных терминалах PowerShell:

```powershell
llama-server -m "C:\models\Qwen3.5-2B-Q4_K_M.gguf" -ngl 99 -c 8192 -np 1 --reasoning off --alias qwen3.5-2b --host 0.0.0.0 --port 8081
llama-server -m "C:\models\bge-m3-Q4_K_M.gguf" -ngl 99 --embeddings --pooling cls --alias bge-m3 --host 0.0.0.0 --port 8082
```

Укажите реальные пути к файлам. Адрес из Docker Desktop — `host.docker.internal`; оба процесса должны быть доступны контейнерам по портам 8081/8082. На видеокарте с 4 GB VRAM оба процесса могут не помещаться одновременно: уменьшите `-ngl` и следите за RAM. Не открывайте незапароленные сервисы в публичную сеть.

**Внимание:** при смене embedding-backend старые и новые векторы нельзя хранить в одном индексе, даже если оба имеют размерность 1024. Сохраните старую БД для отката и создайте отдельную чистую БД/том для повторной индексации 236 PDF; повторите оценку порога и Hit@5.

### Почему Celery вместо BackgroundTasks

`BackgroundTasks` выполняется внутри процесса API; его завершение может прервать работу, а тяжёлая индексация будет расходовать ресурсы API. Celery выполняет задачи отдельным worker и ставит их в очередь Redis. Включены `acks_late` и `task_reject_on_worker_lost`. Повторная доставка возобновляет `pending`/`processing`; блокировка строки БД сериализует запись, а чанки и `completed` фиксируются одной транзакцией. `visibility_timeout=900` больше лимита задачи 600 с. Это покрыто тестами с подменами; реальный аварийный останов воркера пока не проверен.

### Проверка моделей и загрузка корпуса

После запуска двух `llama-server`, из корня проекта:

```powershell
docker compose up -d --build
docker compose exec -T web alembic upgrade head
docker compose exec -T web python -m scripts.check_models
```

`scripts.check_models` проверяет `/health`, отправляет два текста на embedding-сервер и один вопрос генератору. Убедитесь, что вектор имеет размерность 1024. Это реальный сетевой тест; CI использует заглушки и не проверяет модели.

Скачайте `RAG-Multi-Corpus` в отдельную папку на компьютере. Например, при расположении в `C:\data\RAG-Multi-Corpus` скрипт можно запустить в отдельном контейнере, смонтировав только папку `datasets`:

```powershell
docker compose run --rm -v "C:\data\RAG-Multi-Corpus\datasets:/corpus:ro" web python -m scripts.import_pdfs /corpus --url http://web:8000 --dry-run
docker compose run --rm -v "C:\data\RAG-Multi-Corpus\datasets:/corpus:ro" web python -m scripts.import_pdfs /corpus --url http://web:8000 --wait --timeout 7200 --report /app/uploads/corpus_import.json
```

Замените путь `C:\data\RAG-Multi-Corpus\datasets` своим. `--dry-run` только считает PDF и проверяет структуру: всего 236 файлов по пяти компаниям. Обычный запуск отправляет `POST /documents` с JWT компании, ждёт завершения индексации и сохраняет JSON-отчёт в Docker-томе `documents_data` (`/app/uploads/corpus_import.json`). Повторная отправка того же файла в той же компании идемпотентна. **Перед реальной загрузкой обязательно используйте новый пустой PostgreSQL-индекс, не затрагивая старый том.**


При оценке времени ~1,5 минуты из задания учитывайте, что обработка 236 документов зависит от выбранного GPU/CPU-offload и скорости отдельного embedding-сервера. Записывайте фактические значения, не предполагайте выполнение норматива заранее.

## Benchmark retrieval

Понадобятся проиндексированные PDF из RAG-Multi-Corpus и CSV вопросов. При создании индекса передавайте имя компании из CSV в JWT и `POST /documents`. Для PDF из группы CloudWay используйте `CloudWay-24`, как в контрольных запросах скрипта. Дождитесь `completed` для загруженных документов. [PDF корпуса](https://github.com/udayallu/RAG-Multi-Corpus/tree/39071af3f4dd25e59f5c59a6f9b6e8e99cd643b3/datasets), [CSV вопросов](https://raw.githubusercontent.com/udayallu/RAG-Multi-Corpus/39071af3f4dd25e59f5c59a6f9b6e8e99cd643b3/bechmark/bechmark-agentic-references/Dataset%20categories%20-%20queries.csv).

CSV содержит колонки `Enterprise Name`, `Query`, `Supporting Facts`. В последней колонке ожидается JSON-массив объектов с полем `filename`. Скрипт печатает SHA-256 входного CSV, параметры выборки, результаты для разных порогов и Hit@5. С тем же CSV, seed и индексом состав выборки повторяется.

Если CSV лежит только в `/tmp` запущенного worker, сначала сохраните его на хост. Например, `docker compose cp worker:/tmp/rag_benchmark.csv /path/to/rag_benchmark.csv`. Временная папка контейнера не входит в именованный том.

```sh
docker compose cp ./rag_benchmark.csv worker:/tmp/rag_benchmark.csv
docker compose exec -T worker \
  python -m scripts.evaluate_retrieval \
  --benchmark /tmp/rag_benchmark.csv --seed 42 --questions-per-company 20
```

Результаты проверки **ментора** на `Qwen3.5-2B` и `bge-m3`, переданные автором в задании ревью:

| Проверка | Результат ментора |
| --- | --- |
| Индексация | 236 PDF за 43 с, без ошибок |
| Повторная индексация | 236 дублей, ни одного нового чанка |
| Минимальный similarity релевантных вопросов | 0,524 |
| Максимальный similarity нерелевантных вопросов | 0,422 |
| Hit@5 | 77/78 (98,7%) |
| Порог similarity | 0,45 |

Это внешние результаты, а не собственный повторный прогон и не подтверждение нового чанкинга. На этом компьютере модели и корпус отсутствуют; своих чисел индексации и retrieval после исправлений нет. Сохраняйте исходные результаты `evaluate_retrieval --report` для сравнения.

SHA-256 использованного CSV — `58be4e8ebe96147ab63b159b3166357ad905989023956038ed563f9b88c2d24e`. Исходный CSV и документы нужны для повтора этого результата.

### TTFT

```sh
docker compose run --rm -v "./benchmarks:/results" web python -m scripts.measure_ttft \
  --company 'ZX Bank' --questions-file /results/questions.json \
  --warmup-question 'What services does ZX Bank provide?' \
  --server-version 'ACTUAL_LLAMA_SERVER_VERSION' --output /results/ttft_qwen35.csv
```

Подготовьте `questions.json`: JSON-массив минимум из девяти уникальных релевантных вопросов одной компании, количество кратно трём. Скрипт назначает `top_k=2/3/5` по очереди и использует каждый вопрос ровно один раз (минимум три замера на Top-K). Вопрос прогрева отдельный. Релевантность нужно проверить по своему индексу. Вместо `ACTUAL_LLAMA_SERVER_VERSION` укажите вывод установленного `llama-server --version`.

`ttft_ms` включает поиск через Celery, построение промпта и ожидание первого содержательного фрагмента ответа. `llm_ttft_ms` начинается перед вызовом llama-server. Оба замера используют `perf_counter` внутри приложения, без доставки SSE по HTTP. Служебные события, пробелы и reasoning исключены. Токены берутся из `usage` llama-server. Каждый замер, включая неуспешный, сохраняется в CSV; рядом записываются версия/параметры в `.metadata.json` и таблица медиан только успешных замеров в `.md`. Предыдущий CSV не перезаписывается.

Во время TTFT клиент отправляет `cache_prompt=false`, чтобы общий префикс не переиспользовался. Это параметр [API llama-server](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md). Поддержку в установленной сборке нужно проверить по логам обработки промпта; `--cache-ram 0` в новых сборках отключает RAM-cache, но сам по себе не доказывает отсутствие переиспользования KV текущего слота. Инструкция проверки — в [протоколе](docs/verification.md).

Актуальной таблицы TTFT на Qwen3.5-2B пока нет: реальный стенд недоступен. Старый [CSV Ollama](benchmarks/ttft_ollama.csv) сохранён как исторический артефакт с некорректным протоколом повторения вопросов; его числа исключены из итоговой таблицы. Причина первоначального выбора Ollama/qwen3:1.7b с автором не согласована и здесь не предполагается.

## Оставшиеся проверки перед PR

Локальные проверки исправлений: [отчёт от 24.11.2026](benchmarks/verification-2026-11-24.md). Команды для отсутствующего реального стенда: [docs/verification.md](docs/verification.md). Этот список остаётся открытым до фактического прохождения проверок.

- Запустить обе GGUF-модели на реальном компьютере и проверить подключение из Docker.
- Заново проиндексировать все 236 PDF и измерить время обработки, Hit@5 и порог.
- Повторить TTFT для 2/3/5 чанков, каждый замер с новым вопросом.
- Проверить восстановление незавершённых задач при настоящем падении worker (логика повторной доставки покрыта тестами, но не проверялась с реальным Redis).
- Добавить в PR реальный JSON `/ask`, запись SSE и актуальную таблицу измерений.

Подготовка к ответам ментору: [docs/razbor.md](docs/razbor.md).
