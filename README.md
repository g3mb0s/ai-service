# AI service

Асинхронный FastAPI-сервис для чата с OpenAI и генерации упражнений.

## Архитектура

Каждый домен находится в `src/domains/<domain>`:

- `routes.py` — HTTP и создание `AsyncSession` через `Depends`;
- `manager.py` — бизнес-логика, вызовы OpenAI, `flush` и `commit`;
- `service.py` — только чтение и запись через SQLAlchemy;
- `models.py` — ORM-модели;
- `schemas.py` — входные и выходные Pydantic-схемы.

Сервис использует OpenAI Responses API. Для упражнений применяется Structured
Outputs, поэтому ответ модели сразу валидируется Pydantic-схемой.

## Настройка

Обязательные переменные окружения:

```env
OPENAI_API_KEY=sk-...
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_MODEL=gpt-5.6
AI_PROVIDER=openai
USER_DAILY_TOKEN_LIMIT=50000
USER_MAX_OUTPUT_TOKENS_PER_REQUEST=8000
DB_USER=ai
DB_PASSWORD=ai
DB_HOST=localhost
DB_PORT=5435
DB_NAME=ai
JWT_PUBLIC_KEY_URL=http://localhost:8000/auth/public-key
```

`OPENAI_BASE_URL` задаёт хост выбранного провайдера. Конкретный API-контракт
определяется `AI_PROVIDER`: OpenAI использует Responses API, DeepSeek — Chat
Completions и JSON Output.

### Выбор AI-провайдера

Провайдер выбирается через `AI_PROVIDER`:

```env
# OpenAI / ChatGPT
AI_PROVIDER=openai
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_MODEL=gpt-5.6
OPENAI_API_KEY=sk-...
```

```env
# DeepSeek
AI_PROVIDER=deepseek
OPENAI_BASE_URL=https://api.deepseek.com
OPENAI_MODEL=deepseek-v4-flash
OPENAI_API_KEY=sk-...
```

После изменения переменных контейнер необходимо пересоздать:

```bash
docker compose up -d --build --force-recreate ai_service
```

Оба провайдера реализуют общий `AIManager`. Реализация ChatGPT использует
Responses API и Structured Outputs. Реализация DeepSeek использует Chat
Completions, запрашивает JSON Output и валидирует его той же Pydantic-схемой.

Запуск всего окружения из корня репозитория:

```bash
docker compose up --build ai_service
```

API будет доступен на `http://localhost:8003`, Swagger — на
`http://localhost:8003/docs`. Все доменные ручки требуют Bearer JWT из
`auth_service`.

Для пользователей с ролью `user` действует дневной лимит расхода модели,
задаваемый через `USER_DAILY_TOKEN_LIMIT` (по умолчанию `50000`). Учитывается
фактический `total_tokens` всех ответов в чате за календарный день UTC. Чтобы
запрос не превысил остаток квоты, сервис также ограничивает максимальную длину
следующего ответа. Максимум ответа одного пользовательского запроса задаётся
через `USER_MAX_OUTPUT_TOKENS_PER_REQUEST` (по умолчанию `8000`). Роли `manager`
и `admin` этими лимитами не ограничены.

## Основные ручки

- `POST /chat/conversations` — создать диалог;
- `GET /chat/conversations` — получить свои диалоги;
- `GET /chat/conversations/{id}` — получить диалог с историей;
- `POST /chat/conversations/{id}/messages` — отправить сообщение;
- `POST /exercises/generate` — сгенерировать и сохранить упражнения;
- `GET /exercises` — история генераций;
- `GET /exercises/{id}` — конкретная генерация.

Все маршруты `/exercises` доступны только пользователям с ролью `manager` или
`admin`. Для роли `user` API возвращает `403 Forbidden`.

## Как настраивается генерация упражнений

Есть два уровня настройки.

### Что настраивает сервис

Разработчик или DevOps задаёт через окружение:

- `OPENAI_BASE_URL` — адрес API модели;
- `OPENAI_API_KEY` — ключ для этого API;
- `OPENAI_MODEL` — техническое имя модели;
- `AI_PROVIDER` — имя провайдера для аналитики и истории;
- таймауты, retries, системные инструкции и строгую структуру ответа.

Сервис гарантирует английский язык, нужное количество упражнений, правильный тип
и совместимость payload с контрактом `content_service`.

### Что вводит контент-менеджер

Контент-менеджер не выбирает ключ, хост или модель. Он отправляет только параметры
конкретного учебного задания:

```json
{
  "topic": "Present Simple: утвердительные предложения",
  "level": "A1",
  "exercise_type": "fill_gap_choice",
  "count": 5,
  "tags": ["grammar", "present-simple"],
  "extra_instructions": "Используй бытовую лексику и не используй неправильные глаголы"
}
```

Поле `language` не принимается: язык всегда фиксирован как `en`. Поддерживаемые
`exercise_type`: `fill_gap_choice`, `fill_gap_input`, `matching`,
`sentence_from_audio` и `sentence_from_translation`. Для
`sentence_from_audio` дополнительно обязателен `audio_url`.

Для fill-gap текст использует маркеры контракта `content_service`, например
`She {{gap-1}} to school.`. Подчёркивания (`___`) запрещены схемой. Сгенерированные
payload можно напрямую передать в `POST /exercises` или как элементы `items` в
`POST /exercises/bulk` сервиса контента.

Результат сохраняется в БД и возвращается вместе с моделью, провайдером, хостом,
ID ответа и расходом токенов (`input_tokens`, `output_tokens`, `total_tokens`).

Такое разделение позволяет централизованно сменить модель или провайдера без
переобучения контент-менеджеров и без изменения каждого запроса.

Миграции применяются автоматически при старте контейнера. Вручную:

```bash
uv run alembic -c src/alembic.ini upgrade head
```

Тесты:

```bash
uv run pytest
```
