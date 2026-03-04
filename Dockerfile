FROM python:3.12-slim

WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install packages
RUN pip install --no-cache-dir fastapi uvicorn websockets numpy python-multipart

# Install torch separately (CPU only, smaller size)
RUN pip install --no-cache-dir torch==2.2.1+cpu --extra-index-url https://download.pytorch.org/whl/cpu

# Copy entire project
COPY . .

# Tell Python where to find backend module
ENV PYTHONPATH=/app

EXPOSE 8000

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]