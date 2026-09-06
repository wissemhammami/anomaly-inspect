FROM python:3.13-slim

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

COPY src/ ./src/
COPY models/ ./models/
COPY results/ ./results/

EXPOSE 8000

CMD uvicorn src.api.app:app --host 0.0.0.0 --port ${PORT:-8000}