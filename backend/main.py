"""
main.py

Research-grade FastAPI WebSocket inference server.
Architecture: PVDF-NKBT-Eu Biosensor → Whisper Encoder + MLP Classifier → Avatar Actuation
Publication: Nano Energy (Supplementary Note 3)

Run:
    uvicorn main:app --host 0.0.0.0 --port 8000 --reload

The frontend connects to:
    ws://localhost:8000/ws/infer
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from model.load_model import load_model
from inference.pipeline import InferencePipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# App lifecycle — load model once at startup
# ─────────────────────────────────────────────
pipeline: InferencePipeline | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline
    logger.info("Loading model...")
    model = load_model()
    pipeline = InferencePipeline(model=model)
    logger.info("Server ready. Waiting for WebSocket connections.")
    yield
    logger.info("Server shutting down.")


app = FastAPI(
    title="Biosensor Speech Recognition Server",
    description="Real-time WebSocket inference: Whisper Encoder + MLP Classifier",
    version="2.0.0",
    lifespan=lifespan,
)

# Allow the HTML file to connect from any origin (file://, localhost:*, etc.)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────
# Health check
# ─────────────────────────────────────────────
@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model": "WhisperEncoder+MLPClassifier",
        "pipeline_ready": pipeline is not None,
    }

@app.get("/")
async def root():
    return {
        "message": "Silent Speech Avatar Backend Running",
        "status": "ok"
    }


# ─────────────────────────────────────────────
# WebSocket inference endpoint
# ─────────────────────────────────────────────
@app.websocket("/ws/infer")
async def websocket_infer(websocket: WebSocket):
    await websocket.accept()
    client = websocket.client
    logger.info(f"WebSocket connected: {client}")

    try:
        while True:
            # Receive raw audio bytes from the browser
            raw_bytes = await websocket.receive_bytes()
            logger.info(f"Received {len(raw_bytes)} bytes from {client}")

            # Run inference
            result = await pipeline.run(raw_bytes)

            # Send JSON result back to the frontend
            await websocket.send_json(result)

    except WebSocketDisconnect:
        logger.info(f"Client disconnected: {client}")
    except Exception as e:
        logger.error(f"Inference error for {client}: {e}", exc_info=True)
        try:
            await websocket.send_json({"error": str(e)})
        except Exception:
            pass
        await websocket.close(code=1011)