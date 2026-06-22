# 3D FloorPlan — Panduan Lengkap

> Satu file untuk semua instruksi: codebase, CubiCasa5K, instalasi, penyesuaian.

---

## Daftar Isi

1. [Apa Ini? — Gambaran Besar](#1-apa-ini--gambaran-besar)
2. [Codebase & Struktur Project](#2-codebase--struktur-project)
3. [Pipeline Lengkap](#3-pipeline-lengkap)
4. [Apa itu CubiCasa5K?](#4-apa-itu-cubicasa5k)
5. [Instalasi — Local Pipeline](#5-instalasi--local-pipeline)
6. [Instalasi — CubiCasa5K di VPS](#6-instalasi--cubicasa5k-di-vps)
7. [Penyesuaian / Adjustments dari CubiCasa5K Original](#7-penyesuaian--adjustments-dari-cubicasa5k-original)
8. [Cara Pakai — Running Pipeline](#8-cara-pakai--running-pipeline)
9. [3D Viewer](#9-3d-viewer)
10. [Troubleshooting](#10-troubleshooting)
11. [Referensi](#11-referensi)

---

## 1. Apa Ini? — Gambaran Besar

Sistem konversi gambar denah lantai 2D → struktur dinding 3D.

**Dua bagian utama:**

| Bagian | Lokasi | Fungsi |
|--------|--------|--------|
| **ML Server** | VPS `YOUR_VPS_IP:8000` | Segmentasi denah → binary wall mask (CubiCasa5K) |
| **Local Pipeline** | Laptop/PC | wall mask → deteksi dinding → graph topologi → JSON → 3D viewer |

**Filosofi inti:** *Walls are source of truth.* Dinding adalah kebenaran. Ruangan diturunkan dari dinding, bukan sebaliknya.

Pipeline sepenuhnya deterministik (OpenCV). ML hanya untuk segmentasi awal wall mask.

---

## 2. Codebase & Struktur Project

```
3dfloorplan/
├── docs/                          ← Dokumentasi (folder ini)
│   ├── OVERVIEW.md                ← Arsitektur pipeline detail
│   ├── INSTALLATION.md            ← Setup local dependencies
│   ├── CUBICASA.md                ← Setup CubiCasa5K di VPS
│   ├── USAGE.md                   ← Panduan penggunaan lengkap
│   └── README.md                  ← ← Ini: panduan komprehensif
│
├── wallgraph/                     ← ★ Python package utama
│   ├── __init__.py                ← Public API exports
│   ├── models.py                  ← Point, Wall, Door, Window, WallGraph (Pydantic)
│   ├── detector.py                ← WallDetector — ekstraksi dinding (OpenCV)
│   ├── builder.py                 ← WallGraphBuilder — topology cleanup
│   ├── validator.py               ← WallGraphValidator — validasi
│   ├── constants.py               ← Semua threshold/parameter terpusat
│   ├── utils.py                   ← Geometry helpers
│   └── cli.py                     ← CLI entry point
│
├── viewer/
│   └── index.html                 ← Three.js 3D viewer (standalone, tanpa bundler)
│
├── sample/raw/                    ← 15 sample gambar denah untuk testing
├── output/                        ← Wall mask hasil ML API
│
├── running.py                     ← ENTRY POINT UTAMA
├── test_api.py                    ← CLI tester untuk ML API
│
├── pyproject.toml                 ← Python dependencies
├── uv.lock                        ← Lock file (uv)
└── .python-version                ← Python 3.11
```

### Package `wallgraph/` — Komponen Inti

| File | Baris | Fungsi |
|------|-------|--------|
| `detector.py` | ~2092 | WallDetector: thresholding, morfologi, skeleton, Hough, contour, arc detection, door detection |
| `builder.py` | ~476 | WallGraphBuilder: endpoint snap, junction resolve, dedup, filter noise, Layer C bridging |
| `constants.py` | ~218 | Semua magic number: snap distance, min wall length, toleransi sudut, dll |
| `models.py` | ~90 | Data model Pydantic: Point, Wall, Door, Window, WallGraph |
| `validator.py` | ~174 | Validasi topologi: floating wall, duplicate, disconnected component |
| `utils.py` | ~199 | Helper: distance, angle, collinear check |
| `cli.py` | ~87 | CLI: `python -m wallgraph` |

---

## 3. Pipeline Lengkap

```
[Gambar denah lantai 2D]
        │
        ▼ (HTTP POST /predict/wall-mask)
┌─────────────────────────────────────┐
│  VPS: ML Server (CubiCasa5K)        │
│  - Preprocessing (furniture/annotation removal)  │
│  - Inferensi semantic segmentation   │
│  - Ekstraksi channel wall (class 2)  │
│  → Output: binary wall mask PNG     │
└─────────────────────────────────────┘
        │ (white = wall, black = background)
        ▼
┌─────────────────────────────────────┐
│  Local: WallDetector                 │
│  1. Binarize mask                    │
│  2. Morphological close (isi gap)    │
│  3. Morphological open (buang noise)  │
│  4. Skeletonize                      │
│  5. Prune spurs                      │
│  6. Probabilistic Hough Line Transform│
│  7. Snap horizontal/vertical         │
│  8. Merge collinear segments         │
│  9. Merge parallel wall pairs        │
│  10. Arc detection (contour fitting) │
│  11. Door detection (gap scanning)   │
│  → Output: raw wall segments + doors │
└─────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────┐
│  Local: WallGraphBuilder             │
│  1. _snap_endpoints                  │
│  2. _resolve_junctions               │
│  3. _filter_short_walls              │
│  4. _deduplicate                     │
│  5. _snap_endpoints (pass 2)         │
│  6. _filter_noise (stubs)            │
│  7. _close_dangling_endpoints (L-C)  │
│  8. _filter_diagonal_walls           │
│  9. _filter_small_components         │
│  → Output: WallGraph (topologi clean)│
└─────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────┐
│  WallGraphValidator                  │
│  Cek: zero-length, disconnected,     │
│  floating walls, duplicates          │
│  → Output: ValidationReport          │
└─────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────┐
│  3D Viewer (Three.js)               │
│  Straight wall → BoxGeometry         │
│  Arc wall → TubeGeometry             │
│  Bridge wall (Layer C) → warna hijau │
│  OrbitControls: rotate/pan/zoom      │
└─────────────────────────────────────┘
```

---

## 4. Apa itu CubiCasa5K?

CubiCasa5K adalah model deep learning untuk **semantic segmentation denah lantai**. Dilatih pada 5.000 denah dari platform CubiCasa.

### Detail

| Item | Value |
|------|-------|
| **Repo** | https://github.com/CubiCasa/CubiCasa5k |
| **Paper** | *CubiCasa5K: A Dataset and an Improved Multi-Task Model for Floorplan Image Analysis* |
| **Output** | 44 class segmentasi (dinding, pintu, jendela, kamar, dll) |
| **License** | MIT |
| **Model file** | `model_best_val_loss_var.pkl` (checkpoint trained) |
| **Backbone** | ResNet (pretrained `model_1427.pth`) |

### Di project ini, CubiCasa5K dipakai untuk:

1. **Terima gambar denah** raw (dengan furniture, anotasi, dll)
2. **Preprocessing** bersihkan noise (opsional)
3. **Inferensi** semantic segmentation
4. **Ekstraksi channel wall** (class index 2 dari 44 class) → binary mask

Binary mask ini kemudian diproses oleh local pipeline (WallDetector).

### VPS Server Spec

| Item | Value |
|------|-------|
| Host | `YOUR_VPS_IP` |
| OS | Debian 12 |
| CPU | 4 core |
| RAM | 7.8 GB |
| GPU | ❌ CPU only |
| Python | 3.10 |
| Service | FastAPI + uvicorn di port 8000 |
| Root dir | `/opt/floorplan/` |

> **Catatan:** SSH akses ke VPS butuh password atau key. Tidak ada akses publik.

---

## 5. Instalasi — Local Pipeline

### Requirements

- Python 3.11+
- `uv` (recommended) atau `pip`
- Browser modern (Chrome/Firefox/Safari) untuk 3D viewer
- OpenCV runtime (di-include via `opencv-python-headless`)

### Langkah

#### 1. Clone project

```bash
cd ~/projects
git clone <repo-url> 3dfloorplan
cd 3dfloorplan
```

#### 2. Install dependencies

**Dengan `uv` (recommended — lebih cepat):**

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync
```

**Dengan `pip` (alternatif):**

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
pip install requests rich    # untuk test_api.py
```

#### 3. Dependencies utama (auto-install dari pyproject.toml)

| Package | Min versi | Fungsi |
|---------|-----------|--------|
| `opencv-python-headless` | 4.9+ | Image processing, Hough, morfologi |
| `numpy` | 1.26+ | Array operations |
| `networkx` | 3.2+ | Graph topology analysis |
| `pydantic` | 2.0+ | Data models + JSON serialization |
| `shapely` | 2.0+ | Geometry utilities |
| `requests` | any | HTTP ke ML API (opsional) |
| `rich` | any | Pretty console output (opsional) |

#### 4. Verifikasi instalasi

```bash
# Cek CLI
python -m wallgraph --help

# Test deteksi dari sample (tanpa API)
python -m wallgraph sample/raw/image-3.jpg --cli

# Test deteksi dengan mask (kalau sudah ada output mask)
python3 running.py sample/raw/image-3.jpg --mask output/wall_mask_image-3.png --cli
```

#### 5. Jalankan test suite

```bash
pytest tests/ -v
```

---

## 6. Instalasi — CubiCasa5K di VPS

### 6.1. Setup awal

```bash
# SSH ke VPS
ssh root@YOUR_VPS_IP

# Buat direktori dan venv
mkdir -p /opt/floorplan
cd /opt/floorplan
python3 -m venv venv
source venv/bin/activate
```

### 6.2. Clone CubiCasa5K

```bash
git clone https://github.com/CubiCasa/CubiCasa5k.git
cd CubiCasa5k
```

### 6.3. Install dependencies

```bash
# PyTorch CPU-only (VPS tanpa GPU)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Dependencies CubiCasa
pip install opencv-python numpy scipy scikit-image pillow

# FastAPI server
pip install fastapi uvicorn python-multipart
```

> **Catatan:** Package `torch` versi CPU ~200MB, versi GPU ~2GB.

### 6.4. Download model checkpoint

```bash
cd /opt/floorplan/CubiCasa5k

# Install gdown
pip install gdown

# Download dari Google Drive
gdown "https://drive.google.com/uc?id=<FILE_ID>" -O model_best_val_loss_var.pkl
```

> **⚠️ File ID** tergantung link Google Drive resmi CubiCasa5K. Cek repo README untuk link terbaru.
>
> **⚠️ JANGAN** pakai `model_1427.pth` sebagai checkpoint — itu backbone init weights, bukan trained model.

### 6.5. Verifikasi model

```bash
cd /opt/floorplan/CubiCasa5k
python3 -c "
import torch
ckpt = torch.load('model_best_val_loss_var.pkl', map_location='cpu')
print('keys:', list(ckpt.keys()))
print('OK')
"

# Expected: keys: ['model_state', 'optimizer_state', ...]
```

### 6.6. Buat FastAPI wrapper (`app.py`)

Buat file `/opt/floorplan/CubiCasa5k/app.py`:

```python
import io
import time
import numpy as np
import cv2
import torch
from fastapi import FastAPI, File, UploadFile, Query
from fastapi.responses import Response
from floortrans.models import get_model

app = FastAPI(title="FloorPlan ML Server")

# ── Load model ──
device = torch.device('cpu')
ckpt = torch.load('model_best_val_loss_var.pkl', map_location='cpu')
model = get_model('resnet50', 44, True)
model.load_state_dict(ckpt['model_state'])
model.eval()
model.to(device)
print(f"Model loaded. Epoch: {ckpt.get('epoch', '?')}")

# ── Preprocessing ──
def remove_furniture_gray_walls(img_bgr, threshold=100):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)
    result = img_bgr.copy()
    result[mask > 0] = [255, 255, 255]
    return result

def remove_annotations(img, min_area_ratio=0.00005):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)
    img_area = img.shape[0] * img.shape[1]
    min_area = max(50, int(img_area * min_area_ratio))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    keep_mask = np.zeros_like(binary)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            keep_mask[labels == i] = 255
    result = img.copy()
    result[keep_mask == 0] = [255, 255, 255]
    return result

@app.get("/health")
def health():
    return {"status": "ok", "model": "CubiCasa5K", "device": str(device)}

@app.post("/predict/wall-mask")
async def predict_wall_mask(
    file: UploadFile = File(...),
    max_dim: int = Query(1024, description="Max dimension for resize. 0 = no resize"),
    preprocess: str = Query(None, regex="^(furniture|annotation|both)?$"),
    furniture_threshold: int = Query(100, ge=1, le=255),
):
    start = time.time()
    contents = await file.read()
    img_bgr = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
    orig_h, orig_w = img_bgr.shape[:2]

    # Preprocessing
    if preprocess == "furniture":
        img_bgr = remove_furniture_gray_walls(img_bgr, furniture_threshold)
    elif preprocess == "annotation":
        img_bgr = remove_annotations(img_bgr)
    elif preprocess == "both":
        img_bgr = remove_furniture_gray_walls(img_bgr, furniture_threshold)
        img_bgr = remove_annotations(img_bgr)

    # Resize
    if max_dim and max_dim > 0:
        h, w = img_bgr.shape[:2]
        scale = max_dim / max(h, w)
        if scale < 1.0:
            new_w, new_h = int(w * scale), int(h * scale)
            img_bgr = cv2.resize(img_bgr, (new_w, new_h))

    # Model inference
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_f32 = img_rgb.astype(np.float32) / 255.0
    img_norm = 2 * img_f32 - 1
    tensor = torch.tensor(img_norm).permute(2, 0, 1).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(tensor)
    pred = torch.argmax(output['room_type'], dim=1).squeeze(0).cpu().numpy()
    wall_mask = (pred == 2).astype(np.uint8) * 255

    # Resize back ke original size
    if max_dim and max_dim > 0:
        wall_mask = cv2.resize(wall_mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

    _, buf = cv2.imencode('.png', wall_mask)
    elapsed = time.time() - start

    return Response(
        content=buf.tobytes(),
        media_type="image/png",
        headers={
            "X-Inference-Time": f"{elapsed:.3f}",
            "X-Preprocess": preprocess or "none",
            "X-Original-Size": f"{orig_w}x{orig_h}",
        }
    )
```

### 6.7. Jalankan server

```bash
# HARUS dari direktori CubiCasa5k
cd /opt/floorplan/CubiCasa5k
source ../venv/bin/activate

# Testing (foreground)
uvicorn app:app --host 0.0.0.0 --port 8000

# Production (background)
nohup uvicorn app:app --host 0.0.0.0 --port 8000 > /var/log/floorplan.log 2>&1 &
```

> **⚠️ Penting:** Server HARUS dijalankan dari `/opt/floorplan/CubiCasa5k/` karena CubiCasa5K pakai relative path hardcoded untuk load backbone (`floortrans/models/model_1427.pth`). Jalan dari direktori lain → `FileNotFoundError`.

### 6.8. Update server (kalau ada perubahan `app.py`)

```bash
# Local: copy app.py ke VPS
scp app.py root@YOUR_VPS_IP:/opt/floorplan/CubiCasa5k/

# SSH, kill, restart
ssh root@YOUR_VPS_IP
pkill -f "uvicorn"
cd /opt/floorplan/CubiCasa5k
source ../venv/bin/activate
nohup uvicorn app:app --host 0.0.0.0 --port 8000 > /var/log/floorplan.log 2>&1 &
```

### 6.9. Cek log server

```bash
ssh root@YOUR_VPS_IP
tail -f /var/log/floorplan.log
```

### 6.10. Test API dari local

```bash
# Pakai test_api.py
python3 test_api.py -f sample/raw/image-3.jpg

# Dengan preprocessing
python3 test_api.py -f sample/raw/image-4.jpeg --preprocess furniture
python3 test_api.py -f sample/raw/image-13.jpg --preprocess both

# Langsung curl
curl -X POST http://YOUR_VPS_IP:8000/predict/wall-mask \
  -F "file=@sample/raw/image-3.jpg" \
  -o output/wall_mask_test.png
```

---

## 7. Penyesuaian / Adjustments dari CubiCasa5K Original

### 7.1. FastAPI wrapper (`app.py`) — Baru

Repo original hanya punya training/evaluation script. Kami buat `app.py` sebagai HTTP API:
- `GET /health` — health check
- `POST /predict/wall-mask` — inference endpoint
- Auto-resize gambar sebelum inference
- Preprocessing functions untuk noise removal

### 7.2. Working directory constraint — Tidak diubah, hanya didokumentasi

CubiCasa5K pakai **relative path hardcoded** untuk load backbone:
```python
# Di floortrans/models/__init__.py
init_weights('floortrans/models/model_1427.pth')
```
→ Server HARUS jalan dari root CubiCasa5k folder. Kami tidak modifikasi kode CubiCasa5K, hanya dokumentasi constraint ini.

### 7.3. Model loading — Adjustment

Ada dua file model:
- **`model_1427.pth`** — backbone init weights (ResNet pretrained), di-load otomatis oleh CubiCasa5K
- **`model_best_val_loss_var.pkl`** — trained checkpoint (di-download terpisah dari Google Drive)

```python
# SALAH — ini backbone weights, bukan checkpoint
torch.load('floortrans/models/model_1427.pth')  # KeyError: 'model_state'

# BENAR — ini trained checkpoint
torch.load('model_best_val_loss_var.pkl')  # {'model_state': ..., 'epoch': 1427, ...}
```

### 7.4. CPU-only inference — Adjustment

Repo asli asumsikan GPU. Di VPS tanpa GPU, load dengan `map_location='cpu'`:
```python
ckpt = torch.load('model_best_val_loss_var.pkl', map_location='cpu')
model.load_state_dict(ckpt['model_state'])
model.eval()
```

### 7.5. Preprocessing tambahan — Baru

Dua fungsi preprocessing **tidak ada** di CubiCasa5K original, ditambahkan di `app.py`:

#### `remove_furniture_gray_walls(img, threshold=100)`
Untuk denah dengan furniture hitam di atas background abu-abu:
```python
def remove_furniture_gray_walls(img_bgr, threshold=100):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)
    result = img_bgr.copy()
    result[mask > 0] = [255, 255, 255]
    return result
```

#### `remove_annotations(img, min_area_ratio=0.00005)`
Untuk denah dengan banyak anotasi (text, hatching, dimensi):
```python
def remove_annotations(img, min_area_ratio=0.00005):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)
    img_area = img.shape[0] * img.shape[1]
    min_area = max(50, int(img_area * min_area_ratio))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    keep_mask = np.zeros_like(binary)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            keep_mask[labels == i] = 255
    result = img.copy()
    result[keep_mask == 0] = [255, 255, 255]
    return result
```

### 7.6. Ringkasan perbandingan

| Aspek | CubiCasa5K Original | Project Ini |
|-------|---------------------|-------------|
| Interface | CLI training/eval | FastAPI HTTP API |
| Preprocessing | None | Furniture removal, annotation removal |
| Hardware | GPU assumed | CPU-only (map_location cpu) |
| Model loading | Auto via class | Manual `torch.load` + `load_state_dict` |
| Output | 44 class tensor | Binary wall mask PNG |
| Working dir | Any | Must be `/opt/floorplan/CubiCasa5k/` |

---

## 8. Cara Pakai — Running Pipeline

### Quick start

```bash
# Recommended: ML mask → 3D
python3 running.py sample/raw/image-3.jpg --api

# Dengan mask yang sudah ada
python3 running.py sample/raw/image-3.jpg --mask output/wall_mask_image-3.png
```

### `running.py` — Semua opsi

```bash
python3 running.py [IMAGE] [OPTIONS]
```

| Argumen | Default | Fungsi |
|---------|---------|--------|
| `IMAGE` | (interactive picker) | Path gambar denah |
| `--mask PATH` | — | Pakai wall mask yang sudah ada |
| `--api [URL]` | `http://YOUR_VPS_IP:8000` | Fetch mask dari ML API |
| `--preprocess MODE` | — | `furniture`, `annotation`, atau `both` |
| `--cli` | — | Print JSON ke stdout (tanpa browser) |
| `-o PATH` | — | Simpan JSON ke file |
| `--port N` | 8765 | Port HTTP server viewer |
| `--snap N` | 12.0 (mask) / 8.0 (image) | Snap distance endpoint (px) |
| `--min-length N` | 25.0 (mask) / 35.0 (image) | Min wall length (px) |

### Contoh

```bash
# Interactive mode
python3 running.py

# Mask + 3D viewer
python3 running.py sample/raw/image-3.jpg --mask output/wall_mask_image-3.png

# API + preprocess furniture
python3 running.py sample/raw/image-4.jpeg --api --preprocess furniture

# API + both preprocessing
python3 running.py sample/raw/image-13.jpg --api --preprocess both

# Export JSON saja
python3 running.py sample/raw/image-3.jpg --mask output/wall_mask_image-3.png -o graph.json

# Custom port
python3 running.py image.png --api --port 9000
```

### Tuning parameter

**Terlalu banyak dinding palsu:**
```bash
python3 running.py image.png --mask mask.png --min-length 40 --snap 15
```

**Dinding terputus-putus:**
```bash
python3 running.py image.png --mask mask.png --min-length 20
```

### Pilih preprocessing

| Tipe denah | Rekomendasi |
|------------|-------------|
| Denah bersih (line art) | Tidak perlu preprocessing |
| Furniture hitam di atas bg abu-abu | `--preprocess furniture` |
| Banyak text/dimensi/arsiran | `--preprocess annotation` |
| Keduanya | `--preprocess both` |

### Konstanta penting di `wallgraph/constants.py`

| Konstanta | Default | Fungsi |
|-----------|---------|--------|
| `SNAP_DISTANCE` | 8.0 | Max jarak endpoint untuk digabung |
| `GRAPH_MIN_WALL_LENGTH` | 35.0 | Min panjang wall setelah graph build |
| `COLLINEAR_MERGE_GAP` | 15.0 | Max gap antar collinear segment |
| `DANGLING_ENDPOINT_MAX_GAP` | 30.0 | Layer C: max gap yang bisa di-bridge |
| `GRAPH_ANGLE_TOLERANCE` | 0.4 | Filter diagonal wall |
| `MIN_COMPONENT_NODES` | 4 | Min nodes untuk keep component |

---

## 9. 3D Viewer

### Cara load graph

1. **File picker**: klik "Load WallGraph JSON" → pilih file
2. **Drag & drop**: drag `graph.json` ke browser
3. **URL param**: `http://localhost:8765?json=graph.json`

### Kontrol kamera

| Aksi | Kontrol |
|------|---------|
| Orbit (putar) | Left mouse drag |
| Pan (geser) | Right mouse drag |
| Zoom | Scroll wheel |

### Sliders

| Slider | Default | Fungsi |
|--------|---------|--------|
| Wall height | 2.8 m | Tinggi dinding |
| Scale | 0.0100 m/px | 1 pixel = N meter |
| Default thickness | 0.15 m | Tebal dinding default |

### Warna walls

| Warna | Arti |
|-------|------|
| Krem/beige | Dinding interior |
| Abu-abu gelap | Dinding eksterior |
| Abu-abu kecoklatan | Arc wall |
| Hijau | Bridge wall (Layer C gap-fill) |

### Serve manual (tanpa running.py)

```bash
cd viewer
python -m http.server 8765
# → buka http://localhost:8765
```

---

## 10. Troubleshooting

| Error | Solusi |
|-------|--------|
| `ModuleNotFoundError: wallgraph` | `pip install -e .` dari project root |
| Browser tidak terbuka | Buka manual `http://localhost:8765` |
| Port 8765 sudah terpakai | `--port 9000` atau port lain |
| `cv2.ximgproc not found` | Fallback ke iterative morphology (lebih lambat). Atau `pip install opencv-contrib-python-headless` |
| API timeout / connection refused | Cek VPS: `ssh` → `tail -f /var/log/floorplan.log` |
| `FileNotFoundError: model_1427.pth` | Server harus jalan dari `/opt/floorplan/CubiCasa5k/` |
| `KeyError: 'model_state'` | Salah load file: pakai `model_best_val_loss_var.pkl`, bukan `model_1427.pth` |

---

## 11. Referensi

| File | Isi |
|------|-----|
| `docs/OVERVIEW.md` | Arsitektur pipeline detail |
| `docs/INSTALLATION.md` | Setup local dependencies |
| `docs/CUBICASA.md` | Setup CubiCasa5K di VPS (versi lengkap) |
| `docs/USAGE.md` | Panduan penggunaan lengkap |
| `methodlogy.md` | Blueprint optimasi pipeline 12-stage |
| `Plan3D_Wall_First_CAD_Blueprint.md` | Rencana refactor arsitektur |
| `README.md` | README utama project |
