import csv
import json
from pathlib import Path
from typing import cast, Dict, List, Sequence, TypedDict

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


class MetricsResult(TypedDict):
    accuracy: float
    classification_report: str
    confusion_matrix: List[List[int]]
    labels: List[str]


def compute_metrics(y_true: Sequence[str], y_pred: Sequence[str], labels: List[str]) -> MetricsResult:
    accuracy = float(accuracy_score(y_true, y_pred))
    report = cast(
        str,
        classification_report(
            y_true,
            y_pred,
            labels=labels,
            digits=4,
            zero_division=0,
        ),
    )
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    return {
        "accuracy": accuracy,
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "labels": labels,
    }


def save_confusion_matrix_csv(cm: List[List[int]], labels: List[str], path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["predicted\\true"] + labels)
        for label, row in zip(labels, cm):
            writer.writerow([label] + row)


def save_evaluation_rows(rows: List[Dict[str, object]], path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["image_path", "y_true", "y_pred", "correct", "latency_seconds"]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def save_json_report(data: Dict[str, object], path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
