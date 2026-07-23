import time
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware


class TimingMiddleware(BaseHTTPMiddleware):
    """
    Middleware для засекания времени обработки запроса.
    Добавляет время обработки в заголовки ответа.
    """
    
    async def dispatch(self, request: Request, call_next) -> Response:
        start_time = time.time()
        
        response = await call_next(request)
        
        process_time = time.time() - start_time
        
        response.headers["X-Process-Time"] = str(process_time)
        
        return response