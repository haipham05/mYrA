FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt ./
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install --require-hashes \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    -r requirements.txt

RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0t64 libgl1 libxcb1 \
    && rm -rf /var/lib/apt/lists/*

COPY app ./app
COPY alembic ./alembic
COPY alembic.ini ./alembic.ini

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
