FROM python:3.12-slim

RUN apt-get update && apt-get install -y \
    gcc \
    postgresql-client \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
ENV PYTHONPATH=/app/src

COPY pyproject.toml .python-version uv.lock ./

RUN pip install uv

RUN uv sync --frozen --no-dev

COPY run.py ./
COPY src/ ./src/

EXPOSE 8000

CMD ["uv", "run", "--no-sync", "python", "run.py"]
