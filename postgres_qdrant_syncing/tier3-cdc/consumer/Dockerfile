FROM python:3.11-slim

WORKDIR /app

# Copy shared library and consumer
COPY shared /app/shared
COPY tier3-cdc/consumer /app/consumer

# Install dependencies
RUN pip install --no-cache-dir \
    "qdrant-client[fastembed]>=1.14.0" \
    confluent-kafka>=2.3.0 \
    python-dotenv>=1.0.0

ENV PYTHONPATH=/app

CMD ["python", "consumer/main.py"]