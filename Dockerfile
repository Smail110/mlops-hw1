FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src
COPY models ./models

RUN mkdir -p /app/input /app/output

CMD ["python", "-m", "src.main", "--input", "/app/input/test.csv", "--output-dir", "/app/output", "--model", "/app/models/fraud_hash_logreg.json"]
