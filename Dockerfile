FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    SURVEY_DB_PATH=/data/surveys.db

WORKDIR /app

# 先装依赖，利用 Docker 层缓存
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app

RUN mkdir -p /data
VOLUME ["/data"]

EXPOSE 8000

# 单进程 + SQLite WAL；线程内处理请求，WAL 保证读写并发安全
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
