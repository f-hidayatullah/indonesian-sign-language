import argparse
import time
from typing import List

import cv2
import numpy as np

from models.common import DEFAULT_CONFIDENCE
from models.hybrid import HybridPipeline
from models.mediapipe_rf import MediaPipeRFPipeline
from models.yolo_only import YoloOnlyPipeline


DEFAULT_YOLO_MODEL = "runs/detect/train_v4_negative3/weights/best.pt"
DEFAULT_RF_MODEL = "artifacts/rf_model.pkl"


def draw_predictions(frame: np.ndarray, predictions: List[dict]) -> None:
    try:
        for prediction in predictions:
            x1, y1, x2, y2 = prediction["bbox"]
            label = prediction["label"]
            yolo_conf = prediction.get("yolo_conf")
            rf_conf = prediction.get("rf_conf")

            if x1 < 0 or y1 < 0 or x2 >= frame.shape[1] or y2 >= frame.shape[0]:
                continue

            cv2.rectangle(frame, (x1, y1), (x2, y2), (34, 139, 34), 2)
            details = label
            if yolo_conf is not None:
                details += f" YOLO:{yolo_conf:.2f}"
            if rf_conf is not None:
                details += f" RF:{rf_conf:.2f}"

            cv2.putText(
                frame,
                details,
                (x1, max(24, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
            )
    except Exception as e:
        print(f"Error drawing predictions: {e}")


def build_pipeline(name: str, yolo_model: str, rf_model: str, max_hands: int, yolo_conf: float):
    if name == "yolo-only":
        return YoloOnlyPipeline(yolo_model, conf_threshold=yolo_conf)
    if name == "media-pipe-rf":
        return MediaPipeRFPipeline(rf_model, max_hands=max_hands)
    if name == "hybrid":
        return HybridPipeline(yolo_model, rf_model, conf_threshold=yolo_conf)
    raise ValueError(f"Pipeline tidak dikenal: {name}")


def run_camera(pipeline_name: str, yolo_model: str, rf_model: str, source: int, max_hands: int, yolo_conf: float) -> None:
    pipeline = build_pipeline(pipeline_name, yolo_model, rf_model, max_hands, yolo_conf)
    cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError(f"Webcam tidak bisa dibuka: {source}")

    prev_time = time.time()
    frame_count = 0

    try:
        while True:
            success, frame = cap.read()
            if not success:
                break

            start_time = time.time()
            predictions = pipeline.predict(frame, max_hands=max_hands)
            latency = time.time() - start_time

            draw_predictions(frame, predictions)

            fps = 1.0 / max(time.time() - prev_time, 1e-6)
            prev_time = time.time()

            cv2.putText(frame, f"Pipeline: {pipeline_name}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (200, 200, 0), 2)
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (200, 200, 0), 2)
            cv2.putText(frame, f"Latency: {latency*1000:.0f} ms", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (200, 200, 0), 2)

            try:
                cv2.imshow(f"SIBI {pipeline_name}", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
            except Exception as e:
                print(f"Display error: {e}. Running headless mode.")
                break

            frame_count += 1
            if frame_count % 30 == 0:
                print(f"Frame {frame_count}: predictions={len(predictions)}, latency={latency*1000:.0f}ms, fps={fps:.1f}")
    finally:
        cap.release()
        cv2.destroyAllWindows()


def run_image(pipeline_name: str, yolo_model: str, rf_model: str, image_path: str, max_hands: int, yolo_conf: float) -> None:
    pipeline = build_pipeline(pipeline_name, yolo_model, rf_model, max_hands, yolo_conf)
    frame = cv2.imread(image_path)
    if frame is None:
        raise FileNotFoundError(f"Image tidak dapat dibaca: {image_path}")

    predictions = pipeline.predict(frame, max_hands=max_hands)
    draw_predictions(frame, predictions)
    print("Predictions:")
    for item in predictions:
        print(item)

    cv2.imshow(f"SIBI {pipeline_name}", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(description="Jalankan pipeline SIBI secara terpisah.")
    parser.add_argument("--pipeline", default="yolo-only", choices=["yolo-only", "media-pipe-rf", "hybrid"], help="Pipeline yang dijalankan.")
    parser.add_argument("--yolo-model", default=DEFAULT_YOLO_MODEL, help="Path model YOLO.")
    parser.add_argument("--rf-model", default=DEFAULT_RF_MODEL, help="Path model Random Forest.")
    parser.add_argument("--source", default="0", help="Webcam index atau path file image/video.")
    parser.add_argument("--max-hands", type=int, default=2, help="Jumlah tangan maksimal per frame.")
    parser.add_argument("--yolo-confidence", type=float, default=DEFAULT_CONFIDENCE, help="Confidence threshold untuk prediksi YOLO.")
    args = parser.parse_args()

    if args.source.isdigit():
        run_camera(args.pipeline, args.yolo_model, args.rf_model, int(args.source), args.max_hands, args.yolo_confidence)
    else:
        run_image(args.pipeline, args.yolo_model, args.rf_model, args.source, args.max_hands, args.yolo_confidence)


if __name__ == "__main__":
    main()
