FROM python:3.12-slim

# Install ffmpeg (needed for audio decoding in pipeline.py)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Step 1: numpy<2 first to avoid version conflicts
RUN pip install --no-cache-dir "numpy<2"

# Step 2: lightweight packages
RUN pip install --no-cache-dir \
    fastapi==0.110.0 \
    uvicorn==0.29.0 \
    websockets==12.0 \
    python-multipart==0.0.9

# Step 3: torch CPU-only (use --no-deps to avoid pulling cuda/extras)
RUN pip install --no-cache-dir \
    torch==2.2.1+cpu \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    --no-deps

# Step 4: install torch's actual required deps manually (much smaller than full pull)
RUN pip install --no-cache-dir \
    filelock \
    typing-extensions \
    sympy \
    networkx \
    jinja2 \
    fsspec

# Step 5: openai-whisper (no-deps to avoid re-pulling torch/numpy)
RUN pip install --no-cache-dir openai-whisper --no-deps

# Step 6: whisper's own small deps
RUN pip install --no-cache-dir \
    tiktoken \
    tqdm \
    more-itertools \
    transformers

# Copy project files
COPY . .

# Set working directory to backend so model/ and inference/ are found
WORKDIR /app/backend

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]