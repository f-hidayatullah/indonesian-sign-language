import argparse
import csv
import json
import time
from pathlib import Path
from typing import Dict, List

import cv2

from models.hybrid import HybridPipeline
from models.mediapipe_rf import MediaPipeRFPipeline
from models.yolo_only import YoloOnlyPipeline


DEFAULT_DATASET_ROOT = "Sign-Language-Indonesia.v4i.yolov8"
DEFAULT_YOLO_MODEL = "runs/detect/train_v4_negative3/weights/best.pt"
DEFAULT_RF_MODEL = "artifacts/rf_model.pkl"
DEFAULT_OUTPUT_DIR = "artifacts/negative_evaluation"
SPLITS = ["train", "valid", "test"]


def collect_negative_samples(dataset_root: str, split: str) -> List[str]:
    images_dir = Path(dataset_root) / split / "images"
    labels_dir = Path(dataset_root) / split / "labels"
    if not images_dir.exists() or not labels_dir.exists():
        raise FileNotFoundError(f"Dataset split tidak ditemukan: {images_dir}")

    samples: List[str] = []
    for image_path in sorted(images_dir.glob("*.jpg")):
        label_path = labels_dir / f"{image_path.stem}.txt"
        if not label_path.exists():
            continue
        if label_path.read_text(encoding="utf-8").strip() == "":
            samples.append(str(image_path))
    return samples


def build_pipeline(name: str, yolo_model: str, rf_model: str, max_hands: int):
    if name == "yolo-only":
        return YoloOnlyPipeline(yolo_model)
    if name == "media-pipe-rf":
        return MediaPipeRFPipeline(rf_model, max_hands=max_hands)
    if name == "hybrid":
        return HybridPipeline(yolo_model, rf_model)
    raise ValueError(f"Pipeline tidak dikenal: {name}")


def prediction_field(predictions: List[Dict[str, object]], key: str) -> str:
    values: List[str] = []
    for prediction in predictions:
        value = prediction.get(key)
        if value is None:
            values.append("")
        elif isinstance(value, float):
            values.append(f"{value:.4f}")
        else:
            values.append(str(value))
    return "|".join(values)


def evaluate_negative_images(
    pipeline_name: str,
    pipeline,
    dataset_root: str,
    splits: List[str],
    output_dir: str,
    max_hands: int,
) -> None:
    rows: List[Dict[str, object]] = []
    summary: Dict[str, Dict[str, float]] = {}
    total = 0
    total_fp = 0
    latencies: List[float] = []

    print(f"\n[START] Evaluating negative/null images with {pipeline_name}")
    for split in splits:
        split_samples = collect_negative_samples(dataset_root, split)
        split_total = 0
        split_fp = 0
        print(f"[INFO] {split}: found {len(split_samples)} negative/null images")

        for index, image_path in enumerate(split_samples, start=1):
            if index % 25 == 0:
                print(f"[PROGRESS] {pipeline_name} {split}: {index}/{len(split_samples)}")

            frame = cv2.imread(image_path)
            if frame is None:
                continue

            start = time.time()
            predictions = pipeline.predict(frame, max_hands=max_hands)
            latency = time.time() - start
            latencies.append(latency)

            prediction_count = len(predictions)
            is_false_positive = prediction_count > 0
            split_total += 1
            total += 1
            split_fp += int(is_false_positive)
            total_fp += int(is_false_positive)

            rows.append(
                {
                    "split": split,
                    "image_path": image_path,
                    "prediction_count": prediction_count,
                    "is_false_positive": int(is_false_positive),
                    "predicted_labels": prediction_field(predictions, "label"),
                    "yolo_conf": prediction_field(predictions, "yolo_conf"),
                    "rf_conf": prediction_field(predictions, "rf_conf"),
                    "latency_seconds": round(latency, 4),
                }
            )

        fpr = (split_fp / split_total) if split_total else 0.0
        summary[split] = {
            "samples": split_total,
            "false_positive": split_fp,
            "false_positive_rate": fpr,
        }
        print(f"[DONE] {split}: false_positive={split_fp}/{split_total}, fpr={fpr * 100:.2f}%")

    total_fpr = (total_fp / total) if total else 0.0
    report = {
        "pipeline": pipeline_name,
        "dataset_root": dataset_root,
        "splits": splits,
        "samples": total,
        "false_positive": total_fp,
        "false_positive_rate": total_fpr,
        "avg_latency_seconds": float(sum(latencies) / len(latencies)) if latencies else 0.0,
        "per_split": summary,
    }

    output_path = Path(output_dir) / pipeline_name
    output_path.mkdir(parents=True, exist_ok=True)
    csv_path = output_path / "detailed_negative_results.csv"
    json_path = output_path / "negative_report.json"

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "split",
                "image_path",
                "prediction_count",
                "is_false_positive",
                "predicted_labels",
                "yolo_conf",
                "rf_conf",
                "latency_seconds",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"[RESULT] {pipeline_name}: false_positive={total_fp}/{total}, fpr={total_fpr * 100:.2f}%")
    print(f"[OUTPUT] CSV: {csv_path}")
    print(f"[OUTPUT] JSON: {json_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluasi false positive pipeline pada negative/null images berlabel kosong."
    )
    parser.add_argument("--dataset-root", default=DEFAULT_DATASET_ROOT, help="Root folder dataset YOLO format.")
    parser.add_argument("--splits", nargs="*", default=SPLITS, choices=SPLITS, help="Split yang diuji.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Folder output hasil evaluasi.")
    parser.add_argument("--yolo-model", default=DEFAULT_YOLO_MODEL, help="Path model YOLO.")
    parser.add_argument("--rf-model", default=DEFAULT_RF_MODEL, help="Path model Random Forest.")
    parser.add_argument("--max-hands", type=int, default=2, help="Jumlah prediksi maksimal per frame.")
    parser.add_argument(
        "--pipelines",
        nargs="*",
        default=["yolo-only", "hybrid"],
        choices=["yolo-only", "media-pipe-rf", "hybrid"],
        help="Pipeline yang dievaluasi.",
    )
    args = parser.parse_args()

    for pipeline_name in args.pipelines:
        pipeline = build_pipeline(pipeline_name, args.yolo_model, args.rf_model, args.max_hands)
        evaluate_negative_images(
            pipeline_name=pipeline_name,
            pipeline=pipeline,
            dataset_root=args.dataset_root,
            splits=args.splits,
            output_dir=args.output_dir,
            max_hands=args.max_hands,
        )


if __name__ == "__main__":
    main()
