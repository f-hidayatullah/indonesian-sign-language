from typing import Any, List, Optional, Tuple, TypedDict

import cv2
import numpy as np

from .common import (
    CLASS_NAME_FIX,
    DEFAULT_CONFIDENCE,
    DEFAULT_RF_CONFIDENCE,
    DEFAULT_YOLO_IMG_SIZE,
    create_mediapipe_hands,
    expand_bbox,
    is_valid_hand_landmarks,
    landmarks_to_feature_63d,
    load_rf_model,
    load_yolo_model,
    normalize_rf_label,
)


def _resize_for_mediapipe(bgr_img: np.ndarray) -> np.ndarray:
    h, w = bgr_img.shape[:2]
    min_side = min(h, w)
    max_side = max(h, w)
    scale = 1.0
    if min_side < 256:
        scale = 256 / float(min_side)
    elif max_side > 960:
        scale = 960 / float(max_side)
    if abs(scale - 1.0) < 1e-6:
        return bgr_img
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return cv2.resize(bgr_img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)


def _apply_clahe_bgr(bgr_img: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l2 = clahe.apply(l)
    merged = cv2.merge([l2, a, b])
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


def _candidate_frames(bgr_img: np.ndarray) -> List[np.ndarray]:
    resized = _resize_for_mediapipe(bgr_img)
    flipped = cv2.flip(bgr_img, 1)
    flipped_resized = _resize_for_mediapipe(flipped)
    return [
        bgr_img,
        _apply_clahe_bgr(bgr_img),
        resized,
        _apply_clahe_bgr(resized),
        flipped,
        _apply_clahe_bgr(flipped),
        flipped_resized,
        _apply_clahe_bgr(flipped_resized),
    ]


class PipelinePrediction(TypedDict):
    label: str
    yolo_conf: Optional[float]
    rf_conf: Optional[float]
    bbox: List[int]


class HybridPipeline:
    def __init__(
        self,
        yolo_model_path: str,
        rf_model_path: str,
        conf_threshold: float = DEFAULT_CONFIDENCE,
        rf_conf_threshold: float = DEFAULT_RF_CONFIDENCE,
        imgsz: int = DEFAULT_YOLO_IMG_SIZE,
    ):
        self.yolo_model = load_yolo_model(yolo_model_path)
        self.rf_model, self.rf_labels = load_rf_model(rf_model_path)
        if hasattr(self.rf_model, "n_jobs"):
            self.rf_model.n_jobs = 1
        self.hands = create_mediapipe_hands(
            max_num_hands=2,
            min_detection_confidence=0.35,
            min_tracking_confidence=0.35,
        )
        self.fallback_hands = create_mediapipe_hands(
            max_num_hands=2,
            static_image_mode=True,
            model_complexity=1,
            min_detection_confidence=0.20,
            min_tracking_confidence=0.20,
        )
        self.conf_threshold = conf_threshold
        self.rf_conf_threshold = rf_conf_threshold
        self.imgsz = imgsz

    def _resolve_yolo_label(self, box: Any) -> str:
        try:
            class_id = int(box.cls[0])
            names = getattr(self.yolo_model, "names", {})
            if isinstance(names, dict):
                label = str(names.get(class_id, class_id))
            elif isinstance(names, list) and 0 <= class_id < len(names):
                label = str(names[class_id])
            else:
                label = str(class_id)
        except Exception:
            label = "unknown"
        return CLASS_NAME_FIX.get(label, label)

    def _predict_label(self, feature: np.ndarray) -> Tuple[str, Optional[float]]:
        x = feature.reshape(1, -1)
        pred_label = str(self.rf_model.predict(x)[0])
        pred_label = normalize_rf_label(pred_label, self.rf_labels)

        rf_conf: Optional[float] = None
        if hasattr(self.rf_model, "predict_proba"):
            try:
                probs = self.rf_model.predict_proba(x)[0]
                rf_conf = float(np.max(probs))
            except Exception:
                rf_conf = None

        return pred_label, rf_conf

    def predict(self, frame: np.ndarray, max_hands: int = 2) -> List["PipelinePrediction"]:
        results = self.yolo_model.predict(
            frame, conf=self.conf_threshold, imgsz=self.imgsz, verbose=False
        )
        result = results[0]

        predictions: List[PipelinePrediction] = []
        if result.boxes is None:
            return predictions

        h, w = frame.shape[:2]
        for box in result.boxes:
            if box.xyxy is None or box.conf is None:
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            x1, y1, x2, y2 = expand_bbox(x1, y1, x2, y2, w, h, margin=0.25)
            if x2 <= x1 or y2 <= y1:
                continue

            roi = frame[y1:y2, x1:x2]
            if roi.size == 0:
                continue

            hand_result: Any = None
            for candidate in _candidate_frames(roi):
                candidate_result: Any = self.hands.process(cv2.cvtColor(candidate, cv2.COLOR_BGR2RGB))
                if candidate_result.multi_hand_landmarks:
                    hand_result = candidate_result
                    break

            if hand_result is None:
                for candidate in _candidate_frames(roi):
                    candidate_result: Any = self.fallback_hands.process(cv2.cvtColor(candidate, cv2.COLOR_BGR2RGB))
                    if candidate_result.multi_hand_landmarks:
                        hand_result = candidate_result
                        break

            # Jika area deteksi bukan tangan (MediaPipe tidak menemukan landmark di ROI),
            # tolak deteksi (jangan fallback ke label palsu/non-tangan).
            if hand_result is None or not hand_result.multi_hand_landmarks:
                continue

            landmarks = hand_result.multi_hand_landmarks[0].landmark
            if not is_valid_hand_landmarks(landmarks):
                continue

            feat = landmarks_to_feature_63d(landmarks)
            if feat.size != 63:
                continue

            pred_label, rf_conf = self._predict_label(feat)

            # Tolak jika keyakinan Random Forest terlalu rendah
            if rf_conf is None or rf_conf < self.rf_conf_threshold:
                continue

            predictions.append(
                {
                    "label": pred_label,
                    "yolo_conf": float(box.conf[0]) if box.conf is not None else None,
                    "rf_conf": rf_conf,
                    "bbox": [x1, y1, x2, y2],
                }
            )

        predictions = sorted(predictions, key=lambda item: item["yolo_conf"] or 0.0, reverse=True)[:max_hands]
        predictions = sorted(predictions, key=lambda item: item["bbox"][0])
        return predictions
