FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir fastapi uvicorn websockets numpy python-multipart

RUN pip install --no-cache-dir torch==2.2.1+cpu --extra-index-url https://download.pytorch.org/whl/cpu

COPY . .

WORKDIR /app/backend

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]