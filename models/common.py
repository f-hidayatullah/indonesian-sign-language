import base64
import os
import pickle
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from mediapipe.python.solutions import hands as mp_hands
from ultralytics import YOLO

CLASS_NAME_FIX: Dict[str, str] = {"objects": "O"}
LEGACY_RF_LABELS: List[str] = [
    "A",
    "B",
    "C",
    "D",
    "E",
    "F",
    "G",
    "H",
    "I",
    "K",
    "L",
    "M",
    "N",
    "O",
    "P",
    "Q",
    "R",
    "S",
    "T",
    "U",
    "V",
    "W",
    "X",
    "Y",
]
ALPHABET_CLASSES: List[str] = [
    "A",
    "B",
    "C",
    "D",
    "E",
    "F",
    "G",
    "H",
    "I",
    "K",
    "L",
    "M",
    "N",
    "P",
    "Q",
    "R",
    "S",
    "T",
    "U",
    "V",
    "W",
    "X",
    "Y",
    "O",
]
LEGACY_RF_LABEL_FIX: Dict[str, str] = dict(zip(LEGACY_RF_LABELS, ALPHABET_CLASSES))
DEFAULT_YOLO_IMG_SIZE = 640
DEFAULT_CONFIDENCE = 0.50
DEFAULT_RF_CONFIDENCE = 0.45


def clamp_bbox(x1: int, y1: int, x2: int, y2: int, w: int, h: int) -> Tuple[int, int, int, int]:
    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(0, min(x2, w - 1))
    y2 = max(0, min(y2, h - 1))
    return x1, y1, x2, y2


def expand_bbox(x1: int, y1: int, x2: int, y2: int, w: int, h: int, margin: float = 0.25) -> Tuple[int, int, int, int]:
    bw = x2 - x1
    bh = y2 - y1
    mx = int(bw * margin)
    my = int(bh * margin)
    return clamp_bbox(x1 - mx, y1 - my, x2 + mx, y2 + my, w, h)


def load_rf_model(path: str) -> Tuple[Any, Optional[List[str]]]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"RF model tidak ditemukan: {path}")

    with open(path, "rb") as f:
        payload = pickle.load(f)

    if isinstance(payload, dict) and "model" in payload:
        return payload["model"], payload.get("labels")

    return payload, None


def normalize_rf_label(label: str, rf_labels: Optional[List[str]]) -> str:
    fixed_label = CLASS_NAME_FIX.get(label, label)
    if list(rf_labels or []) == LEGACY_RF_LABELS:
        return LEGACY_RF_LABEL_FIX.get(fixed_label, fixed_label)
    return fixed_label


def load_yolo_model(path: str) -> YOLO:
    if not os.path.exists(path):
        raise FileNotFoundError(f"YOLO model tidak ditemukan: {path}")
    return YOLO(path)


def create_mediapipe_hands(
    max_num_hands: int = 2,
    static_image_mode: bool = True,
    model_complexity: int = 0,
    min_detection_confidence: float = 0.35,
    min_tracking_confidence: float = 0.35,
) -> mp_hands.Hands:
    return mp_hands.Hands(
        static_image_mode=static_image_mode,
        max_num_hands=max_num_hands,
        model_complexity=model_complexity,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )


def is_valid_hand_landmarks(landmarks: Any) -> bool:
    """Memeriksa apakah sekumpulan 21 landmark membentuk struktur tangan yang valid dan bukan artefak pakaian."""
    if landmarks is None:
        return False
    try:
        pts = list(landmarks)
        if len(pts) != 21:
            return False

        xs = [float(p.x) for p in pts]
        ys = [float(p.y) for p in pts]

        span_x = max(xs) - min(xs)
        span_y = max(ys) - min(ys)

        # Landmark terlalu kolaps / titik piksel tunggal
        if span_x < 0.03 or span_y < 0.03:
            return False

        # Jarak pergelangan tangan (0) ke pangkal jari tengah (9)
        wrist = pts[0]
        middle_mcp = pts[9]
        palm_dist = ((wrist.x - middle_mcp.x) ** 2 + (wrist.y - middle_mcp.y) ** 2) ** 0.5
        if palm_dist < 0.02 or palm_dist > 0.95:
            return False

        return True
    except Exception:
        return False


def landmarks_to_feature_63d(landmarks: Any) -> np.ndarray:
    features: List[float] = []
    for point in landmarks:
        features.extend([float(point.x), float(point.y), float(point.z)])
    return np.asarray(features, dtype=np.float32)


def landmarks_to_bbox(landmarks: Any, frame_width: int, frame_height: int, margin: float = 0.15) -> Tuple[int, int, int, int]:
    x_coords = [float(point.x) * frame_width for point in landmarks]
    y_coords = [float(point.y) * frame_height for point in landmarks]
    x1 = int(max(0, min(x_coords)))
    y1 = int(max(0, min(y_coords)))
    x2 = int(min(frame_width - 1, max(x_coords)))
    y2 = int(min(frame_height - 1, max(y_coords)))
    width = x2 - x1
    height = y2 - y1
    mx = int(width * margin)
    my = int(height * margin)
    return clamp_bbox(x1 - mx, y1 - my, x2 + mx, y2 + my, frame_width, frame_height)


def decode_base64_image(image_base64: str) -> np.ndarray:
    payload = image_base64
    if "," in payload:
        payload = payload.split(",", 1)[1]

    try:
        image_bytes = base64.b64decode(payload)
    except Exception as exc:
        raise ValueError(f"Base64 tidak valid: {exc}") from exc

    array = np.frombuffer(image_bytes, dtype=np.uint8)
    frame = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("Gagal decode gambar dari base64.")
    return frame
