from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from basic_utils.logger import get_logger
from contextlib import asynccontextmanager
from basic_utils.config import settings
from basic_utils.database import async_engine
from basic_utils.exceptions import (
    AIProviderError,
    DailyTokenLimitError,
    EntityNotFoundError,
)
from basic_utils.openai_client import close_openai_client
from domains.chat.routes import router as chat_router
from domains.exercises.routes import router as exercises_router
from prometheus_fastapi_instrumentator import Instrumentator
from middleware.timing import TimingMiddleware


logger = get_logger("main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Освобождает HTTP- и DB-пулы при остановке приложения."""

    yield
    await close_openai_client()
    await async_engine.dispose()

app = FastAPI(title=settings.app_name,
    description=settings.app_name,
    version=settings.app_version,
    debug=settings.debug,
    lifespan=lifespan,
)

app.add_middleware(TimingMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=settings.cors_allow_credentials,
    allow_methods=settings.allowed_methods,
    allow_headers=settings.allowed_headers,
)


Instrumentator().instrument(app).expose(app)
app.include_router(chat_router)
app.include_router(exercises_router)


@app.exception_handler(EntityNotFoundError)
async def entity_not_found_handler(
    _request: Request, exc: EntityNotFoundError
) -> JSONResponse:
    return JSONResponse(
        status_code=status.HTTP_404_NOT_FOUND,
        content={"detail": str(exc)},
    )


@app.exception_handler(AIProviderError)
async def ai_provider_error_handler(
    _request: Request, exc: AIProviderError
) -> JSONResponse:
    logger.error("AI provider error", {"error": str(exc)})
    return JSONResponse(
        status_code=status.HTTP_502_BAD_GATEWAY,
        content={"detail": str(exc)},
    )


@app.exception_handler(DailyTokenLimitError)
async def daily_token_limit_handler(
    _request: Request, exc: DailyTokenLimitError
) -> JSONResponse:
    return JSONResponse(
        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
        content={"detail": str(exc)},
    )


@app.get("/")
async def index():
    logger.info("index called")
    return {"status": "ok"}
