FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1
ENV MPLBACKEND=Agg

WORKDIR /app

RUN pip install --no-cache-dir uv

COPY backend ./backend

WORKDIR /app/backend

RUN uv sync --frozen --no-dev

CMD ["sh", "-c", "uv run uvicorn src.server:app --host 0.0.0.0 --port ${PORT:-8002}"]
