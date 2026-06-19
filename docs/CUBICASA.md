# CubiCasa5K — ML Wall Segmentation Server

> **Panduan lengkap ada di [README.md](README.md).** File ini versi detail setup VPS CubiCasa5K saja.

## Apa itu CubiCasa5K?

CubiCasa5K adalah model deep learning untuk segmentasi denah lantai. Dilatih pada 5.000 denah dari CubiCasa platform. Output: 44 class segmentasi (dinding, pintu, jendela, kamar, dll).

- **Repo resmi**: https://github.com/CubiCasa/CubiCasa5k
- **Paper**: _CubiCasa5K: A Dataset and an Improved Multi-Task Model for Floorplan Image Analysis_
- **License**: MIT

Model yang digunakan: `model_best_val_loss_var.pkl` (checkpoint terbaik dari training resmi).

---

## Setup VPS

### Spesifikasi server yang digunakan

| Item | Value |
|------|-------|
| Host | `167.172.88.109` |
| OS | Debian 12 |
| CPU | 4 core |
| RAM | 7.8 GB |
| GPU | ❌ CPU only |
| Python | 3.10 |

### Direktori

```
/opt/floorplan/
├── app.py                ← FastAPI server (custom wrapper)
├── CubiCasa5k/           ← repo CubiCasa5K (di-clone dari GitHub)
│   ├── floortrans/
│   │   ├── models/
│   │   │   └── model_1427.pth     ← backbone init weights (BUKAN checkpoint)
│   │   └── ...
│   └── model_best_val_loss_var.pkl ← checkpoint trained model
└── venv/                 ← Python virtual environment
```

---

## Instalasi CubiCasa5K di VPS

### 1. Buat direktori dan venv

```bash
mkdir -p /opt/floorplan
cd /opt/floorplan
python3 -m venv venv
source venv/bin/activate
```

### 2. Clone CubiCasa5K

```bash
git clone https://github.com/CubiCasa/CubiCasa5k.git
cd CubiCasa5k
```

### 3. Install dependencies CubiCasa

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install opencv-python numpy scipy scikit-image pillow
pip install fastapi uvicorn python-multipart
```

> **Note:** Install PyTorch CPU-only karena VPS tidak punya GPU.
> Package `torch` versi CPU lebih kecil (~200MB vs ~2GB untuk GPU).

### 4. Download model checkpoint

Model `model_best_val_loss_var.pkl` harus di-download manual:

```bash
cd /opt/floorplan/CubiCasa5k

# Dengan gdown (Google Drive)
pip install gdown
gdown "https://drive.google.com/uc?id=<FILE_ID>" -O model_best_val_loss_var.pkl
```

> File ID dapat dari Google Drive link resmi CubiCasa5K.
> Kalau link berubah, cek repo README CubiCasa5K untuk link terbaru.

**JANGAN** gunakan `model_1427.pth` sebagai checkpoint — file itu adalah backbone init weights (ResNet pretrained), bukan trained floorplan model. Akan muncul `KeyError: 'model_state'` jika dipakai sebagai checkpoint.

### 5. Verifikasi model loading

```bash
cd /opt/floorplan/CubiCasa5k
python3 -c "
import torch
ckpt = torch.load('model_best_val_loss_var.pkl', map_location='cpu')
print('keys:', list(ckpt.keys()))
print('OK')
"
# Expected output: keys: ['model_state', 'optimizer_state', ...]
```

---

## FastAPI Server (`/opt/floorplan/app.py`)

Custom wrapper di atas CubiCasa5K. Tidak ada di repo resmi — dibuat untuk project ini.

### Endpoint

```
GET  /health          → status server, model info, epoch
POST /predict/wall-mask  → upload gambar → binary wall mask PNG
```

### Query params `/predict/wall-mask`

| Param | Default | Keterangan |
|-------|---------|-----------|
| `max_dim` | 1024 | Resize gambar ke max dimension ini sebelum inference. 0 = no resize |
| `preprocess` | `null` | Mode preprocessing sebelum inference |
| `furniture_threshold` | 100 | Threshold intensitas untuk filter furniture |

### Mode `preprocess`

| Value | Keterangan |
|-------|-----------|
| `furniture` | Hapus furniture gelap (intensitas < threshold) → putihkan. Untuk denah bergaya gray-wall dengan furniture hitam |
| `annotation` | Hapus elemen kecil (text, hatching, dimensi) via connected component size filter |
| `both` | Kedua preprocessing sekaligus |

### Response Headers

| Header | Keterangan |
|--------|-----------|
| `X-Inference-Time` | Waktu inference di server (detik) |
| `X-Preprocess` | Mode preprocessing yang diaplikasikan |
| `X-Original-Size` | Dimensi gambar asli sebelum resize |

### Menjalankan server

```bash
cd /opt/floorplan/CubiCasa5k
source ../venv/bin/activate

# Foreground (untuk testing)
uvicorn app:app --host 0.0.0.0 --port 8000

# Background (production)
nohup uvicorn app:app --host 0.0.0.0 --port 8000 > /var/log/floorplan.log 2>&1 &
```

> **Penting:** Server HARUS dijalankan dari direktori `/opt/floorplan/CubiCasa5k/`
> karena CubiCasa5K menggunakan relative path untuk load model backbone
> (`floortrans/models/model_1427.pth`). Menjalankan dari direktori lain
> akan menyebabkan `FileNotFoundError`.

---

## Cara Kerja Inference

### Preprocessing gambar

```python
def preprocess_image(img):
    # Normalize ke [-1, 1]
    img = img.astype(np.float32) / 255.0
    img = 2 * img - 1
    # Tambah batch dimension
    return torch.tensor(img).permute(2, 0, 1).unsqueeze(0)
```

### Model output → wall mask

CubiCasa5K output: tensor `[1, 44, H, W]` — 44 class probabilitas.

- **Channel 21–32**: wall-related classes
- Untuk wall mask: ambil argmax channel, filter class 2 (wall)

```python
# Simplified extraction
output = model(input_tensor)
pred = torch.argmax(output['room_type'], dim=1)  # atau output format lain
wall_mask = (pred == 2).cpu().numpy().astype(np.uint8) * 255
```

### Auto-resize performance

| Image size | Inference time (CPU) |
|------------|---------------------|
| 512px | ~3s |
| 1024px | ~6s |
| 2560px (tanpa resize) | ~42s |

`max_dim=1024` adalah default yang baik: quality vs speed tradeoff optimal.

---

## Adjustments / Modifikasi dari CubiCasa5K Original

Beberapa hal yang diubah/ditambah dari repo asli CubiCasa5K:

### 1. FastAPI wrapper (`app.py`) — TIDAK ADA di repo asli

Repo asli hanya menyediakan training/evaluation script. Kami tambahkan `app.py` sebagai HTTP API wrapper:

- `GET /health` — health check
- `POST /predict/wall-mask` — endpoint inference
- Auto-resize gambar sebelum inference
- Preprocessing functions untuk noise removal

### 2. Working directory constraint

CubiCasa5K repo menggunakan **relative path hardcoded** untuk load backbone:
```python
# Di dalam floortrans/models/__init__.py atau sejenisnya
init_weights('floortrans/models/model_1427.pth')
```

Ini artinya server HARUS dijalankan dari root CubiCasa5k folder, bukan `/opt/floorplan`. Tidak ada modifikasi yang dilakukan — cukup pastikan `cd` ke folder yang benar sebelum menjalankan uvicorn.

### 3. Model loading adjustment

File `model_1427.pth` adalah **backbone init weights** (ResNet pretrained), bukan trained floorplan checkpoint. CubiCasa5K load ini secara otomatis saat model initialization.

Trained checkpoint yang benar adalah `model_best_val_loss_var.pkl` yang didownload terpisah dari Google Drive.

```python
# SALAH — ini backbone weights, bukan checkpoint
torch.load('floortrans/models/model_1427.pth')  # KeyError: 'model_state'

# BENAR — ini trained checkpoint
torch.load('model_best_val_loss_var.pkl')  # {'model_state': ..., 'epoch': 1427, ...}
```

### 4. CPU-only inference

Repo asli mengasumsikan GPU tersedia. Di VPS tanpa GPU, pastikan load dengan `map_location='cpu'`:

```python
ckpt = torch.load('model_best_val_loss_var.pkl', map_location='cpu')
model.load_state_dict(ckpt['model_state'])
model.eval()
```

### 5. Preprocessing tambahan

Dua fungsi preprocessing ditambahkan di `app.py` (tidak ada di CubiCasa5K asli):

#### `remove_furniture_gray_walls(img, threshold=100)`
Untuk denah yang menggunakan furniture hitam di atas background abu-abu:
```python
def remove_furniture_gray_walls(img_bgr, threshold=100):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)
    result = img_bgr.copy()
    result[mask > 0] = [255, 255, 255]  # putihkan furniture
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

---

## Restart / Update Server

```bash
# Kill server lama
pkill -f "uvicorn"

# Update app.py (kalau ada perubahan)
scp app.py root@167.172.88.109:/opt/floorplan/CubiCasa5k/

# Start ulang
ssh root@167.172.88.109
cd /opt/floorplan/CubiCasa5k
source ../venv/bin/activate
nohup uvicorn app:app --host 0.0.0.0 --port 8000 > /var/log/floorplan.log 2>&1 &
```

## Cek Log

```bash
ssh root@167.172.88.109
tail -f /var/log/floorplan.log
```
