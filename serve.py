import os
from pathlib import Path

import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from utils_mnist import preprocess_canvas_image

MODEL_PATH = os.getenv("MODEL_PATH", "digit_model.keras")
FALLBACK_MODEL_PATH = "model_artifacts/digit_cnn.keras"
WEB_DIR = Path("web")

app = FastAPI(title="Digit Recognition API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model: tf.keras.Model | None = None


@app.on_event("startup")
def load_model() -> None:
    global model
    model_path = Path(MODEL_PATH)
    if not model_path.exists():
        fallback = Path(FALLBACK_MODEL_PATH)
        if fallback.exists():
            model_path = fallback
    if not model_path.exists():
        raise RuntimeError(
            f"Model not found at '{MODEL_PATH}' or '{FALLBACK_MODEL_PATH}'. Train first with: python train.py --data-root ."
        )
    model = tf.keras.models.load_model(model_path)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> dict:
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if not file.content_type or "image" not in file.content_type:
        raise HTTPException(status_code=400, detail="Please upload a valid image")

    image_bytes = await file.read()
    try:
        tensor = preprocess_canvas_image(image_bytes)
    except Exception as ex:
        raise HTTPException(status_code=400, detail=str(ex)) from ex

    probs = model.predict(tensor, verbose=0)[0]
    pred = int(np.argmax(probs))
    conf = float(probs[pred])

    return {
        "predicted_digit": pred,
        "confidence": conf,
        "probabilities": [float(x) for x in probs],
    }


@app.get("/")
def index() -> FileResponse:
    index_path = WEB_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="UI file not found")
    return FileResponse(index_path)


if WEB_DIR.exists():
    app.mount("/web", StaticFiles(directory=WEB_DIR), name="web")
