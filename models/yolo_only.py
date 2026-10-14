from typing import Any, List, Optional, TypedDict

import cv2
from ultralytics import YOLO

from .common import CLASS_NAME_FIX, DEFAULT_CONFIDENCE, DEFAULT_YOLO_IMG_SIZE, load_yolo_model


class PipelinePrediction(TypedDict):
    label: str
    yolo_conf: Optional[float]
    rf_conf: Optional[float]
    bbox: List[int]


class YoloOnlyPipeline:
    def __init__(
        self,
        model_path: str,
        conf_threshold: float = DEFAULT_CONFIDENCE,
        imgsz: int = DEFAULT_YOLO_IMG_SIZE,
    ):
        self.model: YOLO = load_yolo_model(model_path)
        self.conf_threshold = conf_threshold
        self.imgsz = imgsz
        self.names = getattr(self.model, "names", {}) or {}

    def _resolve_label(self, box: Any) -> str:
        label = None
        if hasattr(box, "cls") and box.cls is not None:
            try:
                class_id = int(box.cls[0])
                if isinstance(self.names, dict):
                    label = str(self.names.get(class_id, class_id))
                elif isinstance(self.names, list) and 0 <= class_id < len(self.names):
                    label = str(self.names[class_id])
                else:
                    label = str(class_id)
            except Exception:
                label = str(box.cls[0])

        if label is None:
            label = "unknown"

        return CLASS_NAME_FIX.get(label, label)

    def predict(self, frame: Any, max_hands: int = 2) -> List[PipelinePrediction]:
        results = self.model.predict(
            frame, conf=self.conf_threshold, imgsz=self.imgsz, verbose=False
        )
        result = results[0]

        predictions: List[PipelinePrediction] = []
        if result.boxes is None:
            return predictions

        for box in result.boxes:
            if box.xyxy is None or box.conf is None:
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            if x2 <= x1 or y2 <= y1:
                continue

            predictions.append(
                {
                    "label": self._resolve_label(box),
                    "yolo_conf": float(box.conf[0]) if box.conf is not None else None,
                    "rf_conf": None,
                    "bbox": [x1, y1, x2, y2],
                }
            )

        predictions = sorted(predictions, key=lambda item: item["yolo_conf"] or 0.0, reverse=True)[
            :max_hands
        ]
        predictions = sorted(predictions, key=lambda item: item["bbox"][0])
        return predictions
