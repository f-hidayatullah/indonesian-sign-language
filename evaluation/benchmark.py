import argparse
import csv
import os
import time
from pathlib import Path
from typing import Dict, List

import cv2

from evaluation.metrics import compute_metrics, save_confusion_matrix_csv, save_evaluation_rows, save_json_report
from models.common import ALPHABET_CLASSES
from models.hybrid import HybridPipeline
from models.mediapipe_rf import MediaPipeRFPipeline
from models.yolo_only import YoloOnlyPipeline


def load_yolo_label_file(path: str) -> List[str]:
    labels: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 1:
                continue
            try:
                label_id = int(float(parts[0]))
            except ValueError:
                continue
            if 0 <= label_id < len(ALPHABET_CLASSES):
                labels.append(ALPHABET_CLASSES[label_id])
            else:
                labels.append(str(label_id))
    return labels


def collect_dataset_samples(dataset_root: str, split: str) -> List[Dict[str, str]]:
    images_dir = Path(dataset_root) / split / "images"
    labels_dir = Path(dataset_root) / split / "labels"
    if not images_dir.exists() or not labels_dir.exists():
        raise FileNotFoundError(f"Dataset tidak ditemukan untuk split {split}: {images_dir}")

    samples: List[Dict[str, str]] = []
    for image_path in sorted(images_dir.glob("*.jpg")):
        label_path = labels_dir / f"{image_path.stem}.txt"
        if not label_path.exists():
            continue
        samples.append({"image": str(image_path), "label_path": str(label_path)})
    return samples


def load_ground_truth(label_path: str) -> List[str]:
    gt = load_yolo_label_file(label_path)
    return gt


def evaluate_pipeline(
    pipeline_name: str,
    pipeline,
    dataset_root: str,
    split: str,
    output_dir: str,
    max_hands: int,
) -> None:
    print(f"\n[START] Evaluating {pipeline_name} on {split}...")
    samples = collect_dataset_samples(dataset_root, split)
    print(f"[INFO] Found {len(samples)} samples")
    
    y_true: List[str] = []
    y_pred: List[str] = []
    rows: List[Dict[str, object]] = []
    latencies: List[float] = []

    for idx, sample in enumerate(samples):
        if (idx + 1) % 10 == 0:
            print(f"[PROGRESS] Processing {idx + 1}/{len(samples)}...")
            
        image_path = sample["image"]
        label_path = sample["label_path"]
        frame = cv2.imread(image_path)
        if frame is None:
            continue

        gt_labels = load_ground_truth(label_path)
        if not gt_labels:
            # Negative/null images dievaluasi terpisah dengan evaluation.negative_benchmark.
            continue

        target_label = gt_labels[0]
        start_time = time.time()
        predictions = pipeline.predict(frame, max_hands=max_hands)
        latency = time.time() - start_time
        latencies.append(latency)

        predicted_label = "NO_PRED"
        if predictions:
            predicted_label = predictions[0]["label"]

        y_true.append(target_label)
        y_pred.append(predicted_label)
        rows.append(
            {
                "image_path": image_path,
                "y_true": target_label,
                "y_pred": predicted_label,
                "correct": int(predicted_label == target_label),
                "latency_seconds": round(latency, 4),
            }
        )

    print(f"[INFO] Processed {len(y_true)} valid samples")
    
    metrics = compute_metrics(y_true, y_pred, labels=ALPHABET_CLASSES)
    output_dir_path = Path(output_dir) / pipeline_name / split
    output_dir_path.mkdir(parents=True, exist_ok=True)

    save_json_report(
        {
            "pipeline": pipeline_name,
            "split": split,
            "samples": len(y_true),
            "accuracy": metrics["accuracy"],
            "avg_latency_seconds": float(sum(latencies) / len(latencies)) if latencies else 0.0,
            "labels": metrics["labels"],
            "classification_report": metrics["classification_report"],
        },
        str(output_dir_path / "report.json"),
    )
    save_confusion_matrix_csv(metrics["confusion_matrix"], metrics["labels"], str(output_dir_path / "confusion_matrix.csv"))
    save_evaluation_rows(rows, str(output_dir_path / "detailed_results.csv"))
    print(f"[DONE] {pipeline_name} split={split} samples={len(y_true)} accuracy={metrics['accuracy']:.4f}")
    print(f"[OUTPUT] Results saved to {output_dir_path}")



def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark pipeline pada sampel gesture berlabel. Label kosong/negative images dilewati."
    )
    parser.add_argument("--dataset-root", default="Sign-Language-Indonesia.v4i.yolov8", help="Root folder dataset YOLO format.")
    parser.add_argument("--split", default="valid", choices=["train", "valid", "test"], help="Dataset split to evaluate.")
    parser.add_argument("--output-dir", default="artifacts/evaluation", help="Folder untuk menyimpan hasil evaluasi.")
    parser.add_argument("--yolo-model", default="runs/detect/train_v4_negative3/weights/best.pt", help="Path ke model YOLO.")
    parser.add_argument("--rf-model", default="artifacts/rf_model.pkl", help="Path ke model Random Forest.")
    parser.add_argument("--max-hands", type=int, default=2, help="Jumlah tangan maksimal yang diprediksi per frame.")
    parser.add_argument("--pipelines", nargs="*", default=["yolo-only", "media-pipe-rf", "hybrid"], help="Pipelines yang akan dievaluasi.")
    args = parser.parse_args()

    print(f"[INIT] SIBI Benchmark Evaluation")
    print(f"[INIT] Dataset: {args.dataset_root}, Split: {args.split}")
    print(f"[INIT] Pipelines: {args.pipelines}")

    available = {
        "yolo-only": YoloOnlyPipeline(args.yolo_model),
        "media-pipe-rf": MediaPipeRFPipeline(args.rf_model, max_hands=args.max_hands),
        "hybrid": HybridPipeline(args.yolo_model, args.rf_model),
    }

    for pipeline_name in args.pipelines:
        pipeline = available.get(pipeline_name)
        if pipeline is None:
            print(f"[ERROR] Unknown pipeline: {pipeline_name}")
            continue
        try:
            evaluate_pipeline(
                pipeline_name=pipeline_name,
                pipeline=pipeline,
                dataset_root=args.dataset_root,
                split=args.split,
                output_dir=args.output_dir,
                max_hands=args.max_hands,
            )
        except Exception as e:
            print(f"[ERROR] {pipeline_name} failed: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n[COMPLETE] All evaluations finished")



if __name__ == "__main__":
    main()
