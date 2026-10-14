# Indonesian Sign Language Detection

Sistem deteksi bahasa isyarat Indonesia (SIBI) berbasis Python, YOLOv8,
MediaPipe Hands, dan Random Forest. Sistem menyediakan backend FastAPI,
antarmuka web, serta runner untuk inferensi melalui webcam atau gambar.

## Fitur

- Deteksi dan klasifikasi gestur menggunakan pipeline **YOLO-only**.
- Pipeline pembanding **MediaPipe Hands + Random Forest**.
- Pipeline **hybrid**: YOLO untuk menentukan area tangan, lalu MediaPipe dan
  Random Forest untuk klasifikasi.
- Backend REST API menggunakan FastAPI.
- Inferensi real-time melalui webcam atau file gambar.
- Modul evaluasi benchmark untuk beberapa pipeline.

## Struktur Utama

```text
app.py                 # Backend FastAPI
run_system.py          # Inferensi webcam/gambar melalui CLI
models/                # Implementasi pipeline model
evaluation/            # Modul evaluasi dan metrik
web/                   # Antarmuka web
requirements.txt       # Dependensi Python
SYSTEM_DOCUMENTATION.md
```

## Persyaratan

- Windows 10/11
- Python 3.10 atau yang kompatibel dengan dependensi di `requirements.txt`
- Webcam, jika menjalankan inferensi real-time

## Instalasi

Buat virtual environment dan install dependensi:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Menjalankan Sistem

### Backend web

```powershell
python -m uvicorn app:app --host 0.0.0.0 --port 8000
```

Kemudian buka [http://localhost:8000](http://localhost:8000).

Alternatifnya, gunakan:

```powershell
.\run_backend.bat
```

### Inferensi melalui webcam

```powershell
python run_system.py --pipeline yolo-only --source 0
```

Pipeline yang tersedia:

```text
yolo-only
media-pipe-rf
hybrid
```

Tekan `q` pada jendela kamera untuk berhenti.

## Model dan Dataset

Karena ukuran file model dan dataset, file berikut tidak disertakan dalam
repository ini:

- Dataset pelatihan dan validasi.
- Hasil eksperimen dan folder `runs/`.
- Model Random Forest `artifacts/rf_model.pkl`.
- Model hasil training YOLO pada `runs/detect/.../weights/best.pt`.
- Virtual environment dan arsip proyek.

Dengan demikian, source code dapat dipelajari dan dikembangkan dari repository
ini, tetapi inferensi penuh memerlukan file model yang sesuai. Letakkan kembali
file model pada struktur folder yang dirujuk oleh `app.py` dan `run_system.py`
sebelum menjalankan backend.

Dokumentasi teknis dan konfigurasi sistem tersedia di
[`SYSTEM_DOCUMENTATION.md`](SYSTEM_DOCUMENTATION.md).

## Lisensi

Belum ditentukan.
