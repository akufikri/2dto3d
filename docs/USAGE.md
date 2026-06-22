# Panduan Penggunaan

> **Panduan lengkap ada di [README.md](README.md).** File ini versi detail penggunaan saja.

## Quick Start

```bash
# Recommended pipeline (ML mask → 3D)
python3 running.py sample/raw/image-3.jpg --mask output/wall_mask_image-3.png

# Auto-fetch dari ML server
python3 running.py sample/raw/image-3.jpg --api
```

---

## `running.py` — Entry Point Utama

Single command untuk semua mode:

```
python3 running.py [IMAGE] [OPTIONS]
```

### Arguments

| Argument | Default | Keterangan |
|----------|---------|-----------|
| `IMAGE` | (interactive picker) | Path ke gambar denah lantai |
| `--mask PATH` | — | Gunakan binary wall mask yang sudah ada |
| `--api [URL]` | `http://YOUR_VPS_IP:8000` | Fetch wall mask dari ML server, lalu proses |
| `--preprocess {furniture,annotation,both}` | — | Mode preprocessing untuk `--api` |
| `--cli` | — | Print WallGraph JSON ke stdout (tidak buka browser) |
| `-o PATH` | — | Simpan JSON ke file (tidak buka browser) |
| `--port N` | 8765 | Port HTTP server untuk viewer |
| `--snap N` | 12.0 (mask) / 8.0 (image) | Endpoint snap distance (px) |
| `--min-length N` | 25.0 (mask) / 35.0 (image) | Minimum wall length (px) |

### Contoh penggunaan

```bash
# Interactive — pilih dari daftar sample
python3 running.py

# Dengan mask yang sudah disimpan
python3 running.py sample/raw/image-3.jpg --mask output/wall_mask_image-3.png

# Auto-fetch mask dari API (recommended)
python3 running.py sample/raw/image-3.jpg --api

# API + preprocessing furniture noise
python3 running.py sample/raw/image-4.jpeg --api --preprocess furniture

# API + dua preprocessing sekaligus
python3 running.py sample/raw/image-13.jpg --api --preprocess both

# Export JSON saja
python3 running.py sample/raw/image-3.jpg --mask output/wall_mask_image-3.png -o graph.json

# Print JSON ke terminal
python3 running.py sample/raw/image-3.jpg --mask output/wall_mask_image-3.png --cli

# Custom port dan parameter
python3 running.py image.png --api --port 9000 --snap 15 --min-length 30
```

---

## `test_api.py` — ML Server Tester

Untuk test ML API secara terpisah tanpa membuka 3D viewer.

```bash
# Mode interaktif
python3 test_api.py

# Langsung test file
python3 test_api.py -f sample/raw/image-3.jpg

# Dengan preprocessing
python3 test_api.py -f sample/raw/image-4.jpeg --preprocess furniture
python3 test_api.py -f sample/raw/image-13.jpg --preprocess annotation
python3 test_api.py -f sample/raw/image-13.jpg --preprocess both

# Custom API URL
python3 test_api.py -f image.jpg --url http://my-server:8000
```

Output disimpan ke `output/wall_mask_<nama>[_<preprocess>].png`.

---

## `wallgraph` CLI — Low-level Access

Untuk akses langsung ke detection pipeline tanpa server:

```bash
# Detect walls, print JSON
python -m wallgraph sample/raw/image-3.jpg

# Dengan mask (recommended)
# Saat ini hanya lewat running.py yang support --mask

# Simpan ke file + buat visualisasi
python -m wallgraph sample/raw/image-3.jpg -o walls.json -v debug.png

# Auto-open visualisasi
python -m wallgraph sample/raw/image-3.jpg -v debug.png --open

# Pilih detection method
python -m wallgraph image.jpg --method hough   # skeleton-based Hough
python -m wallgraph image.jpg --method canny   # Canny edges

# Custom parameters
python -m wallgraph image.jpg --snap 10 --min-length 30
```

---

## 3D Viewer (`viewer/index.html`)

Standalone HTML — tidak perlu bundler/npm.

### Cara load graph

1. **File picker**: klik "Load WallGraph JSON" → pilih file `graph.json`
2. **Drag & drop**: drag file `graph.json` ke browser
3. **URL param**: `http://localhost:8765?json=graph.json` (same-origin)

### Kontrol kamera

| Aksi | Kontrol |
|------|---------|
| Orbit (putar) | Left mouse drag |
| Pan (geser) | Right mouse drag |
| Zoom | Scroll wheel |

### Sliders UI

| Slider | Default | Keterangan |
|--------|---------|-----------|
| Wall height | 2.8 m | Tinggi dinding dalam meter |
| Scale | 0.0100 m/px | 1 pixel = N meter. Adjust sesuai skala denah |
| Default thickness | 0.15 m | Tebal dinding jika tidak ada thickness data |

### Warna walls

| Warna | Arti |
|-------|------|
| Krem/beige | Dinding interior biasa |
| Abu-abu gelap | Dinding eksterior (`exterior: true`) |
| Abu-abu kecoklatan | Arc wall (dinding melengkung) |
| Hijau | Bridge wall (gap-fill otomatis dari Layer C) |

### Serve manual (tanpa `running.py`)

```bash
cd viewer
python -m http.server 8765
# → buka http://localhost:8765
# → drag graph.json ke browser
```

---

## Pilih Preprocessing yang Tepat

| Tipe denah | Rekomendasi |
|------------|-------------|
| Denah bersih (line art) | Tidak perlu preprocessing |
| Denah dengan furniture gelap di atas background abu-abu | `--preprocess furniture` |
| Denah dengan banyak text/dimensi/arsiran | `--preprocess annotation` |
| Denah dengan keduanya | `--preprocess both` |

---

## Tuning Parameters

### Kalau terlalu banyak dinding palsu

```bash
# Naikkan minimum length → buang wall pendek
python3 running.py image.png --mask mask.png --min-length 40

# Naikkan snap → endpoint berdekatan lebih agresif digabung
python3 running.py image.png --mask mask.png --snap 15
```

### Kalau dinding terputus-putus

```bash
# Turunkan minimum length → ambil wall lebih pendek
python3 running.py image.png --mask mask.png --min-length 20

# Kalau pakai API, tambah preprocessing annotation
python3 running.py image.png --api --preprocess annotation
```

### Konstanta di `wallgraph/constants.py`

| Konstanta | Default | Fungsi |
|-----------|---------|--------|
| `SNAP_DISTANCE` | 8.0 | Builder: max jarak endpoint untuk digabung |
| `GRAPH_MIN_WALL_LENGTH` | 35.0 | Builder: min panjang wall setelah graph build |
| `COLLINEAR_MERGE_GAP` | 15.0 | Detector: max gap antar collinear segment untuk digabung |
| `DANGLING_ENDPOINT_MAX_GAP` | 30.0 | Layer C: max gap yang bisa di-bridge |
| `GRAPH_ANGLE_TOLERANCE` | 0.4 | Builder: filter wall diagonal (tan dari max angle off-axis) |
| `MIN_COMPONENT_NODES` | 4 | Builder: min nodes untuk keep component |
| `NOISE_LENGTH_RATIO` | 1.5 | Builder: threshold wall noise (stub) |

---

## Format WallGraph JSON

```json
{
  "walls": [
    {
      "id": "W0010_s0",
      "start": {"x": 262.0, "y": 112.0},
      "end":   {"x": 393.0, "y": 113.0},
      "thickness": 0.0,
      "exterior": false,
      "center": null,
      "radius": 0.0,
      "start_angle": 0.0,
      "end_angle": 0.0
    },
    {
      "id": "W0015_arc",
      "start": {"x": 100.0, "y": 200.0},
      "end":   {"x": 150.0, "y": 180.0},
      "thickness": 5.0,
      "exterior": false,
      "center": {"x": 120.0, "y": 220.0},
      "radius": 25.0,
      "start_angle": -1.5708,
      "end_angle": 0.0
    }
  ],
  "doors": [
    {
      "id": "D0001",
      "wall_id": "W0010_s0",
      "offset": 0.0,
      "width": 56.5,
      "center": {"x": 200.0, "y": 113.0},
      "radius": 40.0,
      "start_angle": 0.0,
      "end_angle": 1.5708,
      "door_type": "swing"
    }
  ],
  "windows": []
}
```

**Arc wall** diidentifikasi dengan `center != null && radius > 0`.  
Angles dalam **radians**.

---

## Integrasi ke Aplikasi Lain

### Export JSON dari Python

```python
from wallgraph.detector import WallDetector
from wallgraph.builder import WallGraphBuilder
import json

detector = WallDetector(params={"min_wall_length": 25})
walls, doors = detector.detect_from_mask("wall_mask.png")

builder = WallGraphBuilder(snap_distance=12, min_wall_length=25)
graph = builder.build(walls)
graph.doors = doors

# JSON string
json_str = graph.model_dump_json(indent=2)

# Dict (untuk further processing)
data = json.loads(json_str)
```

### Konsumsi di Three.js / JavaScript

```javascript
const graph = await fetch('graph.json').then(r => r.json());

for (const wall of graph.walls) {
  if (wall.center) {
    // Arc wall — gunakan TubeGeometry
  } else {
    // Straight wall
    const sx = wall.start.x * SCALE;
    const sz = wall.start.y * SCALE;
    const ex = wall.end.x   * SCALE;
    const ez = wall.end.y   * SCALE;
    const len = Math.hypot(ex - sx, ez - sz);

    const geo = new THREE.BoxGeometry(len, WALL_HEIGHT, WALL_THICK);
    const mesh = new THREE.Mesh(geo, material);
    mesh.position.set((sx + ex) / 2, WALL_HEIGHT / 2, (sz + ez) / 2);
    mesh.rotation.y = -Math.atan2(ez - sz, ex - sx);
    scene.add(mesh);
  }
}
```

**Koordinat**: X dan Y dari WallGraph map ke X dan Z di Three.js (floor plan adalah bidang XZ, Y ke atas).
