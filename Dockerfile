FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY pipeline ./pipeline

RUN mkdir -p \
    data/raw \
    data/features \
    data/models \
    data/predictions \
    data/quality

EXPOSE 8501

CMD ["python", "-m", "pipeline.simulator"]