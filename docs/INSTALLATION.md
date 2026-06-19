# Instalasi — Local Pipeline

> **Panduan lengkap ada di [README.md](README.md).** File ini fokus ke instalasi lokal saja.

## Requirements

- Python 3.11+
- `uv` (recommended) atau `pip`
- Browser modern (Chrome/Firefox/Safari) untuk viewer

---

## 1. Clone / Setup Project

```bash
cd ~/projects
git clone <repo-url> 3dfloorplan
cd 3dfloorplan
```

---

## 2. Install Dependencies

### Dengan `uv` (recommended, lebih cepat)

```bash
# Install uv kalau belum ada
curl -LsSf https://astral.sh/uv/install.sh | sh

# Buat venv + install semua dependencies
uv sync
```

### Dengan `pip` (alternatif)

```bash
python3 -m venv .venv
source .venv/bin/activate      # Linux/macOS
# atau: .venv\Scripts\activate  # Windows

pip install -e .
# kalau butuh test_api.py juga:
pip install requests rich
```

### Dependencies utama

| Package | Versi min | Fungsi |
|---------|-----------|--------|
| `opencv-python-headless` | 4.9+ | image processing, Hough, morphology |
| `numpy` | 1.26+ | array ops |
| `networkx` | 3.2+ | graph topology analysis |
| `pydantic` | 2.0+ | data models + JSON serialization |
| `shapely` | 2.0+ | geometry utilities |
| `requests` | any | untuk `test_api.py` dan `--api` mode |
| `rich` | any | untuk `test_api.py` output |

---

## 3. Verifikasi Instalasi

```bash
# Cek wallgraph CLI
python -m wallgraph --help

# Test detect dari sample (tidak perlu API)
python -m wallgraph sample/raw/image-3.jpg --cli
```

---

## 4. Jalankan 3D Viewer

```bash
# Mode interaktif (pilih sample)
python3 running.py

# Langsung dengan file + ML mask (recommended)
python3 running.py sample/raw/image-3.jpg --mask output/wall_mask_image-3.png

# Auto-fetch mask dari ML API lalu tampilkan 3D
python3 running.py sample/raw/image-3.jpg --api

# Print JSON ke stdout (tanpa browser)
python3 running.py sample/raw/image-3.jpg --mask output/wall_mask_image-3.png --cli
```

Browser akan terbuka otomatis di `http://localhost:8765`.

---

## 5. Test API Tester (opsional)

`test_api.py` untuk testing ML server secara terpisah:

```bash
# Install dependencies test_api
pip install requests rich

# Mode interaktif
python3 test_api.py

# Langsung test file
python3 test_api.py -f sample/raw/image-3.jpg

# Dengan preprocessing furniture noise
python3 test_api.py -f sample/raw/image-4.jpeg --preprocess furniture
```

---

## Struktur Output

```
output/
└── wall_mask_<nama>.png     ← binary mask dari ML API (white=wall)

viewer/
└── graph.json               ← WallGraph JSON (auto-generated saat viewer berjalan)
```

---

## Troubleshooting

### `ModuleNotFoundError: wallgraph`
```bash
# Pastikan project root ada di path, atau install editable:
pip install -e .
```

### Browser tidak terbuka otomatis
Buka manual: `http://localhost:8765`

### Port sudah terpakai
```bash
python3 running.py image.png --mask mask.png --port 9000
```

### OpenCV error `cv2.ximgproc not found`
Skeletonize akan fallback ke iterative morphological approach (lebih lambat tapi berfungsi).
Kalau ingin yang lebih cepat:
```bash
pip install opencv-contrib-python-headless
```
