import csv
import os
import pickle
from collections import Counter
from typing import cast

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split


CSV_PATH = r"D:\skripsi-sign-language\artifacts\features_landmark.csv"
ARTIFACT_DIR = r"D:\skripsi-sign-language\artifacts"
MODEL_PATH = os.path.join(ARTIFACT_DIR, "rf_model.pkl")
REPORT_PATH = os.path.join(ARTIFACT_DIR, "rf_report.txt")
CM_PATH = os.path.join(ARTIFACT_DIR, "rf_confusion_matrix.csv")

RANDOM_STATE = 42
TEST_SIZE = 0.20


def load_features(csv_path):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV tidak ditemukan: {csv_path}")

    x_rows = []
    y_rows = []
    feature_names = None

    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)

        # Kolom fitur dimulai dari lm1_x sampai lm21_z
        if reader.fieldnames is None:
            raise ValueError("Header CSV kosong.")
        feature_names = [c for c in reader.fieldnames if c.startswith("lm")]
        if len(feature_names) != 63:
            raise ValueError(f"Jumlah fitur harus 63, ditemukan {len(feature_names)}.")

        for row in reader:
            label = row.get("label", "").strip()
            if not label:
                continue
            try:
                features = [float(row[k]) for k in feature_names]
            except (ValueError, KeyError):
                continue

            x_rows.append(features)
            y_rows.append(label)

    if not x_rows:
        raise ValueError("Tidak ada baris valid di CSV fitur.")

    x = np.asarray(x_rows, dtype=np.float32)
    y = np.asarray(y_rows)
    return x, y, feature_names


def save_confusion_matrix(cm, labels, out_path):
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["predicted\\true"] + list(labels))
        for i, label in enumerate(labels):
            writer.writerow([label] + list(cm[i]))


def main():
    os.makedirs(ARTIFACT_DIR, exist_ok=True)

    x, y, feature_names = load_features(CSV_PATH)
    label_counts = Counter(y.tolist())

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    model = RandomForestClassifier(
        n_estimators=500,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)
    acc = accuracy_score(y_test, y_pred)
    labels = sorted(np.unique(y))
    report = cast(
        str,
        classification_report(y_test, y_pred, labels=labels, digits=4),
    )
    cm = confusion_matrix(y_test, y_pred, labels=labels)

    # Simpan model + metadata penting
    payload = {
        "model": model,
        "feature_names": feature_names,
        "labels": labels,
        "random_state": RANDOM_STATE,
    }
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(payload, f)

    save_confusion_matrix(cm, labels, CM_PATH)

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("Random Forest Training Report\n")
        f.write("=" * 40 + "\n")
        f.write(f"CSV: {CSV_PATH}\n")
        f.write(f"Samples total: {len(y)}\n")
        f.write(f"Train samples: {len(y_train)}\n")
        f.write(f"Test samples: {len(y_test)}\n")
        f.write(f"Test size: {TEST_SIZE}\n")
        f.write(f"Random state: {RANDOM_STATE}\n")
        f.write("\nClass distribution:\n")
        for k in sorted(label_counts.keys()):
            f.write(f"- {k}: {label_counts[k]}\n")
        f.write(f"\nAccuracy: {acc:.4f}\n\n")
        f.write("Classification report:\n")
        f.write(report)
        f.write("\n")

    print("Selesai training Random Forest")
    print(f"Accuracy: {acc:.4f}")
    print(f"Model: {MODEL_PATH}")
    print(f"Report: {REPORT_PATH}")
    print(f"Confusion matrix: {CM_PATH}")


if __name__ == "__main__":
    main()
