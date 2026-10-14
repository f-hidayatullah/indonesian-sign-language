import os
from typing import List, Optional, cast

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from models.hybrid import HybridPipeline
from models.mediapipe_rf import MediaPipeRFPipeline
from models.yolo_only import YoloOnlyPipeline
from models.common import CLASS_NAME_FIX, decode_base64_image


BASE_DIR = os.path.dirname(__file__)
YOLO_MODEL_PATH = os.path.join(BASE_DIR, "runs", "detect", "train_v4_negative3", "weights", "best.pt")
RF_MODEL_PATH = os.path.join(BASE_DIR, "artifacts", "rf_model.pkl")
WEB_DIR = os.path.join(BASE_DIR, "web")
DEFAULT_PIPELINE = "yolo-only"
PIPELINE_CHOICES = {"hybrid", "yolo-only", "media-pipe-rf"}


class PredictRequest(BaseModel):
    image_base64: str
    pipeline: Optional[str] = DEFAULT_PIPELINE
    max_hands: int = 2


class HandPrediction(BaseModel):
    label: str
    yolo_conf: Optional[float]
    rf_conf: Optional[float]
    bbox: List[int]


class PredictResponse(BaseModel):
    phrase: str
    predictions: List[HandPrediction]


app = FastAPI(title="SIBI API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _get_pipeline(name: Optional[str]):
    normalized = (name or DEFAULT_PIPELINE).lower()
    if normalized == "yolo-only":
        return yolo_only_pipeline
    if normalized == "media-pipe-rf":
        return mediapipe_rf_pipeline
    if normalized == "hybrid":
        return hybrid_pipeline
    raise HTTPException(status_code=400, detail=f"Pipeline tidak valid: {name}")


def _predict_pipeline(frame: np.ndarray, pipeline_name: Optional[str], max_hands: int):
    pipeline = _get_pipeline(pipeline_name)
    return pipeline.predict(frame, max_hands=max_hands)


if not os.path.exists(YOLO_MODEL_PATH):
    raise RuntimeError(f"YOLO model tidak ditemukan: {YOLO_MODEL_PATH}")
if not os.path.exists(RF_MODEL_PATH):
    raise RuntimeError(f"RF model tidak ditemukan: {RF_MODEL_PATH}")


yolo_only_pipeline = YoloOnlyPipeline(YOLO_MODEL_PATH)
mediapipe_rf_pipeline = MediaPipeRFPipeline(RF_MODEL_PATH)
hybrid_pipeline = HybridPipeline(YOLO_MODEL_PATH, RF_MODEL_PATH)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/")
def web_index():
    index_path = os.path.join(WEB_DIR, "index.html")
    if not os.path.exists(index_path):
        raise HTTPException(status_code=404, detail="web/index.html tidak ditemukan.")
    return FileResponse(index_path)


@app.get("/app.js")
def web_app_js():
    js_path = os.path.join(WEB_DIR, "app.js")
    if not os.path.exists(js_path):
        raise HTTPException(status_code=404, detail="web/app.js tidak ditemukan.")
    return FileResponse(js_path, media_type="application/javascript")


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    frame = decode_base64_image(req.image_base64)
    raw_predictions = _predict_pipeline(frame, req.pipeline, req.max_hands)
    predictions: List[HandPrediction] = [
        HandPrediction(
            label=str(prediction["label"]),
            yolo_conf=cast(Optional[float], prediction.get("yolo_conf")),
            rf_conf=cast(Optional[float], prediction.get("rf_conf")),
            bbox=cast(List[int], prediction["bbox"]),
        )
        for prediction in raw_predictions
    ]
    phrase = " ".join(prediction.label for prediction in predictions)
    return PredictResponse(phrase=phrase, predictions=predictions)


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run(app, host="127.0.0.1", port=port)
