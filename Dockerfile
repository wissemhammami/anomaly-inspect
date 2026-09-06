FROM python:3.13-slim

WORKDIR /app

COPY requirements-gradio.txt ./
RUN pip install --no-cache-dir -r requirements-gradio.txt --extra-index-url https://download.pytorch.org/whl/cpu

COPY src/ ./src/
COPY models/ ./models/
COPY results/ ./results/
COPY tests/ ./tests/
COPY app.py ./

CMD python app.py