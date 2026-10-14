"""
Script Ekstraksi Fitur Landmark MediaPipe Hands (63 Dimensi)
-----------------------------------------------------------
Fungsi: Mengekstrak 21 titik koordinat landmark (x, y, z) dari citra berlabel gestur.
Catatan: Hanya memproses citra yang memiliki bounding box tangan. Citra negatif/background
kosong tidak diekstrak karena tidak memiliki landmark tangan.
"""
import os
import csv
from typing import Any

import cv2
from mediapipe.python.solutions import hands as mp_hands

# ===== Konfigurasi =====
DATASET_DIR = r"D:\skripsi-sign-language\Sign-Language-Indonesia.v4i.yolov8"
SPLITS = ["train", "valid", "test"]
OUTPUT_CSV = r"D:\skripsi-sign-language\artifacts\features_landmark.csv"
FAILED_CSV = r"D:\skripsi-sign-language\artifacts\failed_samples.csv"

CLASS_NAMES = [
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "K", "L", "M",
    "N", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "O"
]

# Margin ROI terhadap ukuran bbox YOLO
ROI_MARGINS = [0.20, 0.40, 0.60, 0.85]
MIN_ROI_SIDE_FOR_MP = 256
MAX_VARIANT_SIDE = 960

os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

# Header CSV fitur
feature_header = [
    "sample_id", "source_path", "label", "hand_index",
    "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"
]
for i in range(1, 22):
    feature_header += [f"lm{i}_x", f"lm{i}_y", f"lm{i}_z"]

# Header CSV gagal
failed_header = ["source_path", "label", "reason"]

# Detector utama: lebih ketat
hands_main = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    model_complexity=0,
    min_detection_confidence=0.35,
)

# Detector cadangan: lebih longgar
hands_loose = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    model_complexity=0,
    min_detection_confidence=0.15,
)

# Detector terakhir: sangat longgar (hanya untuk fallback akhir)
hands_very_loose = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    model_complexity=0,
    min_detection_confidence=0.05,
)


def clamp_bbox(x1, y1, x2, y2, img_w, img_h):
    x1 = max(0, min(int(x1), img_w - 1))
    y1 = max(0, min(int(y1), img_h - 1))
    x2 = max(0, min(int(x2), img_w - 1))
    y2 = max(0, min(int(y2), img_h - 1))
    return x1, y1, x2, y2


def expand_bbox(x1, y1, x2, y2, img_w, img_h, margin_ratio):
    bw = x2 - x1
    bh = y2 - y1
    mx = bw * margin_ratio
    my = bh * margin_ratio
    return clamp_bbox(x1 - mx, y1 - my, x2 + mx, y2 + my, img_w, img_h)


def center_crop_bbox(img_w, img_h, ratio=0.80):
    crop_w = int(img_w * ratio)
    crop_h = int(img_h * ratio)
    x1 = (img_w - crop_w) // 2
    y1 = (img_h - crop_h) // 2
    x2 = x1 + crop_w
    y2 = y1 + crop_h
    return clamp_bbox(x1, y1, x2, y2, img_w, img_h)


def normalize_size_for_mp(bgr_img):
    h, w = bgr_img.shape[:2]
    min_side = min(h, w)
    max_side = max(h, w)

    scale = 1.0
    if min_side < MIN_ROI_SIDE_FOR_MP:
        scale = MIN_ROI_SIDE_FOR_MP / float(min_side)
    elif max_side > MAX_VARIANT_SIDE:
        scale = MAX_VARIANT_SIDE / float(max_side)

    if abs(scale - 1.0) < 1e-6:
        return bgr_img

    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return cv2.resize(bgr_img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)


def apply_clahe_bgr(bgr_img):
    lab = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l2 = clahe.apply(l)
    merged = cv2.merge([l2, a, b])
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


def image_variants(bgr_img):
    base = normalize_size_for_mp(bgr_img)
    variants = [base]

    # Tingkatkan kontras dan brightness ringan untuk tangan di kondisi gelap
    bright = cv2.convertScaleAbs(base, alpha=1.2, beta=12)
    variants.append(bright)

    # Perbaiki detail lokal untuk memperjelas kontur tangan
    clahe = apply_clahe_bgr(base)
    variants.append(clahe)

    return variants


def extract_features_from_result(result):
    lm = result.multi_hand_landmarks[0].landmark
    features = []
    for p in lm:
        features.extend([float(p.x), float(p.y), float(p.z)])
    return features


def extract_landmarks_with_fallback(bgr_img):
    for variant in image_variants(bgr_img):
        rgb = cv2.cvtColor(variant, cv2.COLOR_BGR2RGB)

        result_main: Any = hands_main.process(rgb)
        if result_main.multi_hand_landmarks:
            return extract_features_from_result(result_main)

        result_loose: Any = hands_loose.process(rgb)
        if result_loose.multi_hand_landmarks:
            return extract_features_from_result(result_loose)

        result_very_loose: Any = hands_very_loose.process(rgb)
        if result_very_loose.multi_hand_landmarks:
            return extract_features_from_result(result_very_loose)

        # Coba orientasi mirror (flip horizontal) untuk kasus tangan yang gagal pada orientasi asli
        flipped = cv2.flip(variant, 1)
        flipped_rgb = cv2.cvtColor(flipped, cv2.COLOR_BGR2RGB)

        result_main_flip: Any = hands_main.process(flipped_rgb)
        if result_main_flip.multi_hand_landmarks:
            return extract_features_from_result(result_main_flip)

        result_loose_flip: Any = hands_loose.process(flipped_rgb)
        if result_loose_flip.multi_hand_landmarks:
            return extract_features_from_result(result_loose_flip)

        result_very_loose_flip: Any = hands_very_loose.process(flipped_rgb)
        if result_very_loose_flip.multi_hand_landmarks:
            return extract_features_from_result(result_very_loose_flip)

    return None


sample_id = 0
saved = 0
failed = 0
fallback_center_saved = 0
fallback_full_saved = 0

with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f_out, open(
    FAILED_CSV, "w", newline="", encoding="utf-8"
) as f_fail:
    writer = csv.writer(f_out)
    writer.writerow(feature_header)

    failed_writer = csv.writer(f_fail)
    failed_writer.writerow(failed_header)

    for split in SPLITS:
        img_dir = os.path.join(DATASET_DIR, split, "images")
        lbl_dir = os.path.join(DATASET_DIR, split, "labels")

        for img_name in os.listdir(img_dir):
            if not img_name.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            img_path = os.path.join(img_dir, img_name)
            lbl_path = os.path.join(lbl_dir, os.path.splitext(img_name)[0] + ".txt")
            if not os.path.exists(lbl_path):
                continue

            img = cv2.imread(img_path)
            if img is None:
                continue

            h, w = img.shape[:2]
            rel_path = os.path.relpath(img_path, DATASET_DIR)

            with open(lbl_path, "r", encoding="utf-8") as lf:
                lines = [x.strip() for x in lf.readlines() if x.strip()]

            for line in lines:
                parts = line.split()
                if len(parts) != 5:
                    failed += 1
                    failed_writer.writerow([rel_path, "-", "invalid_label_row"])
                    continue

                cls_id = int(float(parts[0]))
                cx, cy, bw, bh = map(float, parts[1:])

                x1 = int((cx - bw / 2) * w)
                y1 = int((cy - bh / 2) * h)
                x2 = int((cx + bw / 2) * w)
                y2 = int((cy + bh / 2) * h)
                x1, y1, x2, y2 = clamp_bbox(x1, y1, x2, y2, w, h)

                label = CLASS_NAMES[cls_id] if 0 <= cls_id < len(CLASS_NAMES) else str(cls_id)

                if x2 <= x1 or y2 <= y1:
                    failed += 1
                    failed_writer.writerow([rel_path, label, "invalid_bbox"])
                    continue

                features = None
                used_bbox = (x1, y1, x2, y2)

                # 1) Coba ROI dengan margin bertingkat
                for margin in ROI_MARGINS:
                    rx1, ry1, rx2, ry2 = expand_bbox(x1, y1, x2, y2, w, h, margin)
                    if rx2 <= rx1 or ry2 <= ry1:
                        continue

                    roi = img[ry1:ry2, rx1:rx2]
                    features = extract_landmarks_with_fallback(roi)
                    if features is not None:
                        used_bbox = (rx1, ry1, rx2, ry2)
                        break

                # 2) Fallback center-crop bertingkat
                if features is None:
                    for ratio in (0.90, 0.80, 0.70):
                        cx1, cy1, cx2, cy2 = center_crop_bbox(w, h, ratio=ratio)
                        center_roi = img[cy1:cy2, cx1:cx2]
                        features = extract_landmarks_with_fallback(center_roi)
                        if features is not None:
                            used_bbox = (cx1, cy1, cx2, cy2)
                            fallback_center_saved += 1
                            break

                # 3) Fallback full image
                if features is None:
                    features = extract_landmarks_with_fallback(img)
                    if features is not None:
                        used_bbox = (0, 0, w - 1, h - 1)
                        fallback_full_saved += 1

                if features is None:
                    failed += 1
                    failed_writer.writerow([rel_path, label, "mediapipe_not_found"])
                    continue

                sample_id += 1
                ux1, uy1, ux2, uy2 = used_bbox
                row = [
                    sample_id,
                    rel_path,
                    label,
                    0,
                    ux1,
                    uy1,
                    ux2,
                    uy2,
                ] + features
                writer.writerow(row)
                saved += 1

hands_main.close()
hands_loose.close()
hands_very_loose.close()

print(
    f"Selesai. saved={saved}, failed={failed}, "
    f"fallback_center_saved={fallback_center_saved}, fallback_full_saved={fallback_full_saved}, "
    f"output={OUTPUT_CSV}, failed_log={FAILED_CSV}"
)
