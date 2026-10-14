from typing import Any, List, Optional, Tuple, TypedDict

import cv2
import numpy as np

from .common import (
    DEFAULT_RF_CONFIDENCE,
    create_mediapipe_hands,
    is_valid_hand_landmarks,
    landmarks_to_bbox,
    landmarks_to_feature_63d,
    load_rf_model,
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


def _candidate_frames(bgr_img: np.ndarray) -> List[Tuple[np.ndarray, bool]]:
    resized = _resize_for_mediapipe(bgr_img)
    flipped = cv2.flip(bgr_img, 1)
    flipped_resized = _resize_for_mediapipe(flipped)
    return [
        (bgr_img, False),
        (_apply_clahe_bgr(bgr_img), False),
        (resized, False),
        (_apply_clahe_bgr(resized), False),
        (flipped, True),
        (_apply_clahe_bgr(flipped), True),
        (flipped_resized, True),
        (_apply_clahe_bgr(flipped_resized), True),
    ]


def _mirror_bbox(bbox: Tuple[int, int, int, int], frame_width: int) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox
    return frame_width - 1 - x2, y1, frame_width - 1 - x1, y2


class PipelinePrediction(TypedDict):
    label: str
    yolo_conf: Optional[float]
    rf_conf: Optional[float]
    bbox: List[int]


class MediaPipeRFPipeline:
    def __init__(
        self,
        rf_model_path: str,
        max_hands: int = 2,
        rf_conf_threshold: float = DEFAULT_RF_CONFIDENCE,
    ):
        self.rf_model, self.rf_labels = load_rf_model(rf_model_path)
        if hasattr(self.rf_model, "n_jobs"):
            self.rf_model.n_jobs = 1
        self.hands = create_mediapipe_hands(
            max_num_hands=max_hands,
            min_detection_confidence=0.35,
            min_tracking_confidence=0.35,
        )
        self.fallback_hands = create_mediapipe_hands(
            max_num_hands=max_hands,
            static_image_mode=True,
            model_complexity=1,
            min_detection_confidence=0.20,
            min_tracking_confidence=0.20,
        )
        self.max_hands = max_hands
        self.rf_conf_threshold = rf_conf_threshold

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

    def predict(self, frame: np.ndarray, max_hands: int = 2) -> List[PipelinePrediction]:
        result: Any = None
        is_flipped = False
        for candidate, candidate_flipped in _candidate_frames(frame):
            candidate_result: Any = self.hands.process(cv2.cvtColor(candidate, cv2.COLOR_BGR2RGB))
            if candidate_result.multi_hand_landmarks:
                result = candidate_result
                is_flipped = candidate_flipped
                break

        if result is None:
            for candidate, candidate_flipped in _candidate_frames(frame):
                candidate_result: Any = self.fallback_hands.process(cv2.cvtColor(candidate, cv2.COLOR_BGR2RGB))
                if candidate_result.multi_hand_landmarks:
                    result = candidate_result
                    is_flipped = candidate_flipped
                    break

        predictions: List[PipelinePrediction] = []
        if result is None or not result.multi_hand_landmarks:
            return predictions

        h, w = frame.shape[:2]
        for landmarks in result.multi_hand_landmarks:
            # Validasi geometri morfologi tangan
            if not is_valid_hand_landmarks(landmarks.landmark):
                continue

            feat = landmarks_to_feature_63d(landmarks.landmark)
            if feat.size != 63:
                continue

            pred_label, rf_conf = self._predict_label(feat)

            # Tolak jika keyakinan Random Forest di bawah threshold (mencegah salah tebak pada baju/non-tangan)
            if rf_conf is None or rf_conf < self.rf_conf_threshold:
                continue

            bbox = landmarks_to_bbox(landmarks.landmark, w, h, margin=0.12)
            if is_flipped:
                bbox = _mirror_bbox(bbox, w)
            predictions.append(
                {
                    "label": pred_label,
                    "yolo_conf": None,
                    "rf_conf": rf_conf,
                    "bbox": [bbox[0], bbox[1], bbox[2], bbox[3]],
                }
            )

        predictions = sorted(predictions, key=lambda item: item["rf_conf"] or 0.0, reverse=True)[:max_hands]
        predictions = sorted(predictions, key=lambda item: item["bbox"][0])
        return predictions
