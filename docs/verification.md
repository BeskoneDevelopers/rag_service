# Проверка исправлений ревью

На 24.11.2026 локальные модели и корпус отсутствуют; автор подтвердил, что скачивать их не нужно из-за ограниченного места. Docker Engine также не был доступен. Ниже команды для стенда автора/ментора, а не выполненные замеры. Автоматические тесты используют явно синтетические данные. Итог локальных команд сохраняется в `benchmarks/verification-2026-11-24.md`.

## Подготовка стенда

Сохраните старую БД/том для отката. Для полного повторного прогона используйте отдельный Compose project с новыми томами (`docker compose -p rag-review ...`) и свободными портами; не удаляйте прежние тома. Все команды ниже используют обычный `docker compose`: при выборе отдельного проекта добавляйте `-p rag-review` **к каждой** команде. Не смешивайте старые embeddings с новой моделью.

Заполните локальный `.env` по `.env.example`, сгенерируйте собственный `JWT_SECRET`. Пример рассчитан на Compose: БД `db:5432`, Redis `redis:6379`, llama-server на `host.docker.internal`. Для API вне Docker используйте `localhost:5433` для БД и отдельно доступный Redis; Compose не публикует Redis наружу. Не публикуйте `.env`, JWT, корпус и GGUF.

```powershell
docker compose config --quiet
docker compose up -d --build
docker compose exec -T web alembic upgrade head
docker compose exec -T web python -m scripts.check_models
```

`check_models` должен подтвердить обе модели и embedding размерности 1024. Запишите GPU/CPU, RAM/VRAM, offload (`-ngl`), контекст, квантизацию и команды запуска моделей в отчёт стенда.

## Prompt cache и TTFT

Сначала зафиксируйте установленную сборку. Не переносите поддержку параметров из текущей документации на неизвестную версию автоматически.

```powershell
llama-server --version | Tee-Object benchmarks/llama-server-version.txt
llama-server --help | Select-String 'cache|reasoning|predict'
```

В [текущем API llama.cpp](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md) `cache_prompt=false` отключает повторное использование KV общего префикса. `measure_ttft` передаёт его явно; обычные запросы сохраняют прежнее поведение cache. В сборках, где есть `--cache-ram 0`, можно дополнительно отключить RAM-cache при запуске. Это отдельный механизм от KV текущего слота. `t_max_predict_ms` не заменяет абсолютный таймаут приложения: у него дополнительные условия срабатывания.

Для установленной версии проверьте исходники/документацию соответствующего commit/build и логи llama-server: измеряемые запросы должны обрабатывать полный промпт, а не только новый суффикс. Сохраните логи. Если сборка игнорирует `cache_prompt`, используйте поддерживаемое этой сборкой очищение слота или отдельный свежий процесс на каждый замер. Такой режим и исключение загрузки модели из TTFT нужно описать отдельно. Без проверки логов нельзя объявлять замеры свободными от cache.

В `benchmarks/questions.json` поместите JSON-массив минимум из 9 различных релевантных вопросов **одной** проиндексированной компании, количество кратно 3. Скрипт использует порядок Top-K `2,3,5,2,3,5,...`; вопросы между группами и с прогревом не повторяются. Проверяйте сопоставимую длину и сложность вопросов: разные вопросы тоже влияют на TTFT.

```powershell
$reviewServerVersion = (Get-Content benchmarks/llama-server-version.txt -Raw).Trim()
docker compose run --rm -v "${PWD}/benchmarks:/results" web python -m scripts.measure_ttft --company 'ZX Bank' --questions-file /results/questions.json --warmup-question 'What services does ZX Bank provide?' --server-version "$reviewServerVersion" --output /results/ttft_qwen35.csv
```

Прогрев должен иметь релевантный контекст и завершиться успешно. Он не входит в CSV замеров, но его параметры/результат сохраняются в metadata. CSV записывается и сбрасывается после каждого запроса; ошибки не превращаются в нулевое время. В таблицу `.md` включаются только `status=ok`. Для отчёта нужны минимум три успешных замера каждого Top-K; при сбоях проведите новый прогон с новыми вопросами и новым именем CSV. Сохраните все попытки. Разрешённые aliases моделей не доказывают содержимое GGUF — подтвердите файлы командами запуска и логами сервера.

Перенесите в README созданную `.md`-таблицу только после успешного прогона на Qwen3.5-2B + bge-m3, со ссылками на CSV, metadata и логи. Исторический CSV Ollama для этого не подходит.

## Индексация и retrieval

Используйте реальный путь к корпусу вместо `C:\data\RAG-Multi-Corpus\datasets`.

```powershell
docker compose run --rm -v 'C:\data\RAG-Multi-Corpus\datasets:/corpus:ro' web python -m scripts.import_pdfs /corpus --url http://web:8000 --dry-run
docker compose run --rm -v 'C:\data\RAG-Multi-Corpus\datasets:/corpus:ro' -v "${PWD}/benchmarks:/results" web python -m scripts.import_pdfs /corpus --url http://web:8000 --wait --report /results/corpus_import.json
docker compose exec -T worker python -c "from sqlalchemy import select,func; from app.models import Chunk; from app.worker_db import sessionlocal; s=sessionlocal(); print(s.scalar(select(func.count(Chunk.id)))); s.close()" | Tee-Object benchmarks/chunks-before-duplicate.txt
docker compose run --rm -v 'C:\data\RAG-Multi-Corpus\datasets:/corpus:ro' -v "${PWD}/benchmarks:/results" web python -m scripts.import_pdfs /corpus --url http://web:8000 --wait --report /results/corpus_duplicate.json
docker compose exec -T worker python -c "from sqlalchemy import select,func; from app.models import Chunk; from app.worker_db import sessionlocal; s=sessionlocal(); print(s.scalar(select(func.count(Chunk.id)))); s.close()" | Tee-Object benchmarks/chunks-after-duplicate.txt
```

Второй отчёт должен содержать 236 `duplicate=true`, статусы `completed`; число чанков до/после должно совпасть. Отчёт импортера разделяет время HTTP-загрузки и ожидания индексации. Полное время — их сумма, не только ожидание после загрузки.

Для существующего индекса DOCX/Markdown сначала просмотрите список затрагиваемых документов. Команда не пересчитывает PDF, не меняет document ID или SHA-256. `--apply` ставит существующие документы в очередь; старые чанки заменяются транзакционно после получения новых embeddings. `pending`/`processing` пропускаются. `--filename` необязателен: без него выбираются все DOCX/Markdown компании.

```powershell
docker compose exec -T worker python -m scripts.reindex_documents --company 'ZX Bank' --filename 'Awards & Recognitions.docx'
docker compose exec -T worker python -m scripts.reindex_documents --company 'ZX Bank' --filename 'Awards & Recognitions.docx' --apply
```

Проверьте реальную компанию документа перед запуском. Дождитесь `completed` по `/documents/{id}`. Повторите вопрос про Awards, убедитесь, что в контексте/чанках есть заголовок и связанные абзацы, а в sources указан правильный filename/section. Тестовый DOCX из pytest не заменяет проверку настоящего `Awards & Recognitions.docx`.

CSV benchmark должен соответствовать документам индекса. SHA-256 печатается и записывается в JSON.

```powershell
docker compose run --rm -v "${PWD}/rag_benchmark.csv:/input/questions.csv:ro" -v "${PWD}/benchmarks:/results" web python -m scripts.evaluate_retrieval --benchmark /input/questions.csv --seed 42 --questions-per-company 20 --report /results/retrieval.json | Tee-Object benchmarks/retrieval.txt
```

## Настоящие ответы API

После индексации выполните запрос к той же компании. Скрипт сохраняет байты ответов, не создаёт текст ответа вручную и не сохраняет JWT. Имя выходной папки должно быть новым. Если API вернёт ошибку/пустые sources или SSE закончится без `sources`, команда завершится неуспешно; частичная запись останется для диагностики.

```powershell
docker compose run --rm -v "${PWD}/benchmarks:/results" web python -m scripts.capture_responses --company 'ZX Bank' --question 'What awards did ZX Bank Ltd. win in 2023?' --top-k 5 --output /results/responses-awards
```

Проверьте `request.json`, `ask.json`, `ask.sse` и возможность публикации содержимого перед коммитом. Лишь после этого добавляйте ссылки на собранные файлы в README.

Вопрос регрессии: `Describe in detail all programs, courses and admission requirements of the School of Engineering.` Выполните его обычным и потоковым запросом к компании университета. Запишите время: генерация ограничена 60 с/512 токенами; поиск Celery имеет отдельный таймаут. Ожидаемы 504/финальный `error` по времени и 502/`error` при усечении. После окончания/отмены запрос к модели должен снова приниматься. Проверьте также конкурирующий запрос (503) и отключение клиента до/после первого токена. Закрытие upstream ограничено ещё 5 с, Redis-команда освобождения — 3 с; при недоступном Redis ключ исчезает по TTL 180 с. Нельзя обещать исчезновение ключа при любом сетевом сбое немедленно.

## Аварийное завершение Celery

Проверку выполняйте на отдельном стенде без чужих задач. Нужен достаточно долгий документ, чтобы успеть увидеть `processing` **до** завершения. Сохраните ID/хеш, число его чанков и журнал.

```powershell
docker compose logs --since 5m worker | Tee-Object benchmarks/worker-before-kill.txt
docker compose kill -s SIGKILL worker
docker compose up -d worker
```

Это останов всего контейнера: Redis возвращает неподтверждённую задачу после `visibility_timeout` (900 с по умолчанию) и периодической проверки. Это не мгновенное восстановление. Отдельно можно завершить только дочерний prefork-процесс в контейнере, оставив родительский worker живым; `task_reject_on_worker_lost` управляет повторной постановкой в этом случае. Запишите выбранный способ, время остановки/восстановления и фактический статус. Убедитесь, что задача не успела завершиться до SIGKILL — иначе тест не проверяет восстановление.

После повторной доставки ожидаются `processing` → `completed`, один набор уникальных `(document_id, chunk_index)` и то же число чанков при дополнительной доставке завершённого документа. Проверьте SQL-счётчики и sources, сохраните логи worker после рестарта. Если статус остался `processing`, задача потерялась или появились дубли — проверка не пройдена. Повторите сценарий для `pending` и для завершения после коммита чанков, но до ACK.

`task_time_limit=600` относится к prefork worker в Linux/Compose. На Windows solo pool hard timeout не даёт эквивалентных гарантий. Детерминированно падающий по hard limit документ может повторяться из-за `reject_on_worker_lost`; исправьте причину или остановите задачу вручную. Transactional outbox и автоматический поиск потерянных pending-записей в это ревью не добавлены.

Настройки опираются на документацию [Redis visibility timeout](https://docs.celeryq.dev/en/stable/getting-started/backends-and-brokers/redis.html#visibility-timeout) и [Celery tasks](https://docs.celeryq.dev/en/stable/userguide/tasks.html). Наличие конфигурации и unit-тестов не означает, что описанный реальный crash-тест выполнен.
