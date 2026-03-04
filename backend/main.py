import logging
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from model.load_model import load_model
from inference.pipeline import InferencePipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Biosensor Speech Recognition Server",
    description="Real-time WebSocket inference: Whisper Encoder + MLP Classifier",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

pipeline = None


def get_pipeline():
    global pipeline

    if pipeline is None:
        logger.info("Loading ML model for first request...")
        model = load_model()
        pipeline = InferencePipeline(model=model)
        logger.info("Model loaded successfully.")

    return pipeline


@app.get("/")
async def root():
    return {
        "message": "Silent Speech Avatar Backend Running",
        "status": "ok"
    }


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "pipeline_ready": pipeline is not None
    }


@app.websocket("/ws/infer")
async def websocket_infer(websocket: WebSocket):

    await websocket.accept()
    client = websocket.client
    logger.info(f"WebSocket connected: {client}")

    pipeline_instance = get_pipeline()

    try:
        while True:

            raw_bytes = await websocket.receive_bytes()

            logger.info(f"Received {len(raw_bytes)} bytes")

            result = await pipeline_instance.run(raw_bytes)

            await websocket.send_json(result)

    except WebSocketDisconnect:
        logger.info("Client disconnected")

    except Exception as e:

        logger.error(f"Inference error: {e}")

        try:
            await websocket.send_json({"error": str(e)})
        except Exception:
            pass

        await websocket.close(code=1011)