# Dokumentasi Sistem SIBI Modular

## Status Final Konfigurasi Sistem

- **Dataset Final:** `Sign-Language-Indonesia.v4i.yolov8` (dan evaluasi benchmark pada split validasi)
- **Model YOLO Final:** `runs/detect/train_v4_negative3/weights/best.pt` (fallback: `runs/detect/train/weights/best.pt`)
- **Model Random Forest:** `artifacts/rf_model.pkl` (dilatih dari 3.855 sampel fitur landmark 63D)
- **Pipeline Default (Web & CLI):** `yolo-only`
- **Pemisahan Pengujian:**
  1. **Evaluasi Akurasi Gestur (24 Kelas):** Menggunakan `evaluation.benchmark` (mengabaikan citra kosong/tanpa label gestur).
  2. **Evaluasi False Positive (Citra Negatif/Null):** Menggunakan `evaluation.negative_benchmark` (menguji ketahanan model pada 600 citra latar belakang tanpa tangan).

---

## Struktur Sistem

Sistem SIBI terdiri dari 3 pipeline modular independen:

```
models/
├── common.py           # Utilitas bersama (load model, threshold, validasi landmark)
├── yolo_only.py        # Pipeline YOLOv8-only (deteksi + klasifikasi langsung)
├── mediapipe_rf.py     # Pipeline MediaPipe Hands + Random Forest
├── hybrid.py           # Pipeline Hybrid (YOLO ROI -> MediaPipe -> Random Forest)
└── __init__.py

evaluation/
├── metrics.py              # Perhitungan Confusion Matrix, Accuracy, F1-Macro
├── benchmark.py            # Evaluasi benchmark akurasi gestur (24 kelas)
├── negative_benchmark.py   # Evaluasi false positive citra negatif/background
└── __init__.py

app.py                  # Backend FastAPI REST endpoint (/predict, /health)
run_system.py           # CLI runner untuk inferensi real-time webcam/gambar
web/                    # Antarmuka web (index.html, app.js)
```

---

## Cara Menjalankan Sistem

### 1. Menjalankan Server Web & Backend

Gunakan file batch yang disediakan:
```cmd
run_backend.bat
```
Lalu buka browser di: `http://localhost:8000`

Atau via terminal:
```powershell
.\.venv\Scripts\python.exe -m uvicorn app:app --host 0.0.0.0 --port 8000 --log-level info
```

---

### 2. Inferensi Real-time via CLI / Webcam

```powershell
# 1. YOLO-only (Default Pipeline)
.\.venv\Scripts\python.exe run_system.py --pipeline yolo-only --source 0

# 2. MediaPipe + Random Forest
.\.venv\Scripts\python.exe run_system.py --pipeline media-pipe-rf --source 0

# 3. Hybrid Pipeline
.\.venv\Scripts\python.exe run_system.py --pipeline hybrid --source 0

# Dari file gambar
.\.venv\Scripts\python.exe run_system.py --pipeline yolo-only --source "path/to/image.jpg"
```

---

### 3. Evaluasi Benchmark

```powershell
# Evaluasi semua pipeline pada dataset validasi (akurasi gestur 24 kelas)
.\.venv\Scripts\python.exe -m evaluation.benchmark --split valid --pipelines yolo-only media-pipe-rf hybrid

# Evaluasi ketahanan terhadap citra negatif/background kosong
.\.venv\Scripts\python.exe -m evaluation.negative_benchmark
```

---

## Deskripsi 3 Pipeline

### A. YOLOv8-only (Pipeline Utama / Default)
- **Alur:** Citra Kamera → YOLOv8 Model → Deteksi Kotak Tangan & Klasifikasi Huruf Langsung.
- **Karakteristik:** Sangat cepat, end-to-end, akurasi tertinggi pada 24 kelas alfabet SIBI statis.

### B. MediaPipe + Random Forest (Baseline Machine Learning)
- **Alur:** Citra Kamera → MediaPipe Hands (ekstraksi 21 koordinat landmark 3D = 63 fitur) → Random Forest Classifier.
- **Karakteristik:** Berbasis geometri tangan, ringan, tidak melakukan deteksi bounding box objek.

### C. Hybrid (Kombinasi YOLO + MediaPipe + RF)
- **Alur:** Citra Kamera → YOLOv8 ROI Localization → Crop Area Tangan → MediaPipe Hands → Random Forest Classifier.
- **Karakteristik:** Memfokuskan ekstraksi landmark pada area tangan yang dilokalisasi oleh YOLOv8.

---

## Catatan Ilmiah & Integritas Data

1. **Dataset Gestur vs Citra Negatif:**  
   Citra berlabel digunakan untuk mengukur *Accuracy*, *Precision*, *Recall*, dan *F1-Macro* 24 kelas. Citra negatif (tanpa label/background) diuji secara terpisah untuk mengukur rasio penolakan (*false positive rate*).
2. **Pelatihan Random Forest:**  
   Random Forest dilatih khusus pada 63 fitur landmark tangan (`rf_report.txt` mencatat pelatihan dari 3.855 sampel dengan akurasi 75,10%). **Catatan integritas:** `artifacts/features_landmark.csv` dan `artifacts/failed_samples.csv` saat ini berukuran 0 byte (ter-truncate belakangan oleh percobaan re-run `extract_features_csv.py` yang tidak selesai) dan `extract_features_csv.py`/`train_random_forest.py` sudah diedit setelah tanggal training model (`rf_model.pkl`/`rf_report.txt` = 14 Jul 2026). Karena repo ini tidak berada di bawah git, versi dataset (v2 vs v4) yang benar-benar menghasilkan `rf_model.pkl` saat ini **tidak dapat diverifikasi dari isi script sekarang**. Gunakan `rf_report.txt` sebagai satu-satunya bukti sah pelatihan RF, dan jangan klaim RF sudah dilatih ulang dengan dataset v4 tanpa training run baru yang terverifikasi.
3. **Hasil Benchmark — perhatikan versi model per folder, jangan disamaratakan:**
   - `artifacts/evaluation_comparison/` (termasuk salinan `for_claude_updated` dan `for's claude`): dihasilkan dengan **model lama** `runs/detect/train/weights/best.pt` pada **dataset `Sign-Language-Indonesia.v2i.yolov8`** split `valid` (lihat header `PIPELINE_COMPARISON_RESULTS.md`). Angka akurasi `yolo-only` 99,43% di sini **bukan** angka model final.
   - `artifacts/evaluation_v4_negative/{yolo-only,media-pipe-rf,hybrid}/test/report.json`: hasil akurasi gestur 24 kelas dengan **model final** `train_v4_negative3` pada **dataset v4i**, split `test` (527 sampel). Ini adalah angka model final yang seharusnya dikutip: `yolo-only` 98,86%, `hybrid` 46,30%, `media-pipe-rf` 28,65%. (Nama foldernya menyesatkan — isinya bukan uji negative image, melainkan akurasi gestur split test.)
   - Uji false positive citra negatif untuk model final (model `train_v4_negative3`, dataset v4i, 600 citra negatif per pipeline): `artifacts/negative_evaluation/{yolo-only,media-pipe-rf,hybrid}/negative_report.json` — dihasilkan resmi via `python -m evaluation.negative_benchmark --pipelines yolo-only media-pipe-rf hybrid`. Hasil: **FPR 0,00% (0/600)** untuk ketiga pipeline. `artifacts/negative_test_hybrid_v4.csv` adalah versi ad-hoc sebelumnya untuk hybrid (hasil sama, FPR 0%). `artifacts/old_model_negative_test_yolo_only.csv` adalah uji FP **model lama** (FPR 11%), jangan dicampur sebagai angka model final.
