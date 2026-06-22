#!/usr/bin/env python3
"""
3D Floorplan Backend API
========================
Exposes the wall-extraction engine as a REST API.
Door/window detection is intentionally left to an external LLM caller.

Endpoints
---------
GET  /health                     — liveness probe
POST /extract/walls              — image or pre-computed mask → wall shapes JSON
POST /merge/openings             — wall shapes JSON + LLM openings → final THREE.js JSON

Flow
----
  1. Client uploads floorplan image to POST /extract/walls
     → receives { type:"shape", shapes:[...], image_size:[w,h] }

  2. Client sends image to external LLM for door/window detection
     → LLM returns structured Opening objects (x, y, width, depth, wall_axis, kind)

  3. Client POSTs { graph: <step-1-result>, openings: <step-2-result> } to POST /merge/openings
     → receives merged JSON ready for THREE.js viewer
"""

from __future__ import annotations

import io
import json
import logging
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Literal

import requests as http_requests

import numpy as np
import cv2
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, Response, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from wallgraph.mask_to_shape import extract_wall_shapes, _normalize_thickness

# ── App ────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="3D Floorplan Engine",
    description="Wall extraction backend. Door/window detection delegated to caller's LLM.",
    version="0.2.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Schemas ────────────────────────────────────────────────────────────────────


class Opening(BaseModel):
    """A single door or window detected by the external LLM."""

    id: str = Field(default="", description="Optional identifier, e.g. 'D0', 'W1'")
    kind: Literal["door", "window"] = Field(
        description="Opening type — 'door' or 'window'"
    )
    x: float = Field(description="Center X in original image pixels")
    y: float = Field(description="Center Y in original image pixels")
    width: float = Field(description="Opening span along the wall (pixels)")
    depth: float = Field(
        default=10.0,
        description="Wall thickness at this opening (pixels). Estimate if unknown.",
    )
    wall_axis: Literal["h", "v"] = Field(
        description=(
            "'h' = horizontal wall (opening visible on north/south faces), "
            "'v' = vertical wall (opening on east/west faces)"
        )
    )


class MergeRequest(BaseModel):
    """Merge wall graph with LLM-provided openings."""

    graph: dict = Field(description="Wall shapes JSON from POST /extract/walls")
    openings: list[Opening] = Field(
        default=[],
        description="Openings detected by the external LLM",
    )


class ExtractParams(BaseModel):
    """Pipeline tuning params (all optional, defaults match best known config)."""

    epsilon: float = Field(default=4.0, description="Polygon simplification (px)")
    close_px: int = Field(default=8, description="Morphological closing radius (px)")
    smooth_blur: int = Field(default=5, description="Gaussian blur before contour (px)")
    dilate_px: int = Field(default=0, description="Extra dilation after close (px)")
    snap_axes: bool = Field(default=True, description="Snap diagonal edges to H/V")
    snap_tol_deg: float = Field(default=25.0, description="Snap tolerance (degrees)")
    min_area: int = Field(default=2000, description="Min contour area (px²)")
    min_extent: int = Field(default=50, description="Min bounding-box side (px)")
    notch_depth: float = Field(default=0.0, description="Notch removal depth (px)")
    uniform_thickness: int = Field(
        default=6, description="Skeleton re-dilation for uniform walls (px, 0=off)"
    )


# ── ML mask API ───────────────────────────────────────────────────────────────

ML_MASK_API = "http://localhost:8000"
_ml_api_available: bool | None = None  # cached after first check

log = logging.getLogger("api")


def _try_ml_mask(image_bytes: bytes, filename: str) -> np.ndarray | None:
    """Try to get a clean wall mask from the ML mask API.

    Returns binary mask (white=wall) or None if ML API unavailable.
    """
    global _ml_api_available
    if _ml_api_available is False:
        return None

    try:
        resp = http_requests.post(
            f"{ML_MASK_API}/predict/wall-mask",
            files={"file": (filename, image_bytes)},
            params={"max_dim": 1024},
            timeout=30,
        )
        if resp.status_code == 200:
            _ml_api_available = True
            arr = np.frombuffer(resp.content, dtype=np.uint8)
            mask = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
                log.info("ML mask API returned clean wall mask")
                return binary
        else:
            log.warning("ML mask API error %d", resp.status_code)
    except Exception:
        _ml_api_available = False
        log.info("ML mask API not available — falling back to threshold")
    return None


# ── Helpers ────────────────────────────────────────────────────────────────────


def _decode_upload(data: bytes) -> np.ndarray:
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Cannot decode uploaded file as image.",
        )
    return img


def _to_binary_mask(img: np.ndarray, is_raw_floorplan: bool = False) -> np.ndarray:
    """Convert any image to a binary (0/255) mask (white=wall).

    For raw floorplan images (dark walls on white background):
      Auto-invert so dark walls → white (wall), white background → black (empty).
    For pre-computed wall masks (white=wall): no inversion needed.
    """
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Auto-detect: if image is predominantly white (mean > 128), walls are dark → invert
    thresh_type = cv2.THRESH_BINARY_INV if (is_raw_floorplan or img.mean() > 128) else cv2.THRESH_BINARY
    _, mask = cv2.threshold(img, 127, 255, thresh_type)

    return mask


def _clean_threshold_mask(gray: np.ndarray) -> np.ndarray:
    """Build a clean wall-only mask from a grayscale floorplan image.

    Uses adaptive threshold + connected component filtering + skeleton
    normalization to extract walls while removing furniture/text/annotations.

    Adaptive threshold handles varying line thickness better than global
    threshold — walls have high local contrast regardless of furniture.
    """
    # 1. Adaptive threshold: better at distinguishing structural lines
    amask = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 51, 10,
    )

    # 2. Connected component filter: keep only the main wall mass
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    closed = cv2.morphologyEx(amask, cv2.MORPH_CLOSE, k, iterations=2)

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)
    if n_labels <= 1:
        return amask

    areas = [(i, stats[i, cv2.CC_STAT_AREA]) for i in range(1, n_labels)]
    areas.sort(key=lambda x: x[1], reverse=True)
    largest_area = areas[0][1]
    threshold = largest_area * 0.10

    clean = np.zeros_like(amask)
    for label_id, area in areas:
        if area >= threshold:
            clean[labels == label_id] = amask[labels == label_id]

    kept = sum(1 for _, a in areas if a >= threshold)
    log.info(
        "Threshold mask cleanup: %d components → kept %d (threshold=%d px²)",
        len(areas), kept, int(threshold),
    )

    # 3. Skeleton normalization: uniform wall thickness, merges at junctions
    clean = _normalize_thickness(clean, half_px=4)

    return clean


async def _resolve_mask(image: UploadFile) -> tuple[np.ndarray, bool]:
    """Resolve a binary wall mask (white=wall) from an uploaded floorplan image.

    Tries the ML mask API first, falls back to adaptive-threshold cleanup.
    Returns (mask, has_clean_mask) — shared by /extract/skeleton and /extract/walls
    so both endpoints agree on what "the skeleton" is.
    """
    img_bytes = await image.read()
    input_img = _decode_upload(img_bytes)

    binary_mask = _try_ml_mask(img_bytes, image.filename or "upload.png")
    if binary_mask is not None:
        return binary_mask, True

    gray = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY) if input_img.ndim == 3 else input_img
    return _clean_threshold_mask(gray), False


# ── Routes ─────────────────────────────────────────────────────────────────────


@app.get("/health", tags=["meta"])
def health() -> dict:
    return {"status": "ok", "version": app.version}


@app.post("/extract/skeleton", tags=["pipeline"])
async def extract_skeleton(
    image: UploadFile = File(..., description="Floorplan image (PNG/JPG)"),
) -> Response:
    """
    Step 2 of the pipeline: convert a raw floorplan photo/sketch into a clean
    black-and-white wall-only mask (skeleton), for the caller to preview/confirm
    before paying for the (slower) vectorization pass in POST /extract/walls.

    Returns a PNG image (white=wall, black=empty).
    """
    binary_mask, _ = await _resolve_mask(image)
    ok, buf = cv2.imencode(".png", binary_mask)
    if not ok:
        raise HTTPException(status_code=500, detail="Failed to encode skeleton mask as PNG.")
    return Response(content=buf.tobytes(), media_type="image/png")


@app.post("/extract/walls", tags=["pipeline"])
async def extract_walls(
    image: UploadFile = File(..., description="Floorplan image (PNG/JPG)"),
    mask: UploadFile = File(
        default=None,
        description=(
            "Pre-computed binary wall mask PNG (white=wall). "
            "If provided, 'image' is only used for its filename."
        ),
    ),
    params: str = Form(
        default="{}",
        description="JSON string of ExtractParams fields to override defaults.",
    ),
) -> JSONResponse:
    """
    Extract wall shapes from a floorplan image or pre-computed mask.

    Returns a `shape` JSON compatible with the THREE.js viewer:
    ```json
    {
      "type": "shape",
      "shapes": [{ "outer": [[x,y],...], "holes": [[[x,y],...], ...] }],
      "image_size": [width, height]
    }
    ```

    Pass `params` as a JSON string to override pipeline defaults:
    ```
    params='{"epsilon":4,"close_px":8,"smooth_blur":5,"snap_tol_deg":25}'
    ```
    """
    t0 = time.perf_counter()

    # Parse params
    try:
        raw_params = json.loads(params)
        p = ExtractParams(**raw_params)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Invalid params JSON: {e}")

    # Load image bytes (always read for saving to output)
    img_bytes = await image.read()
    input_img = _decode_upload(img_bytes)

    # Load mask: explicit mask > ML mask API > threshold fallback
    has_clean_mask = False
    if mask is not None:
        mask_bytes = await mask.read()
        mask_img = _decode_upload(mask_bytes)
        binary_mask = _to_binary_mask(mask_img, is_raw_floorplan=False)
        has_clean_mask = True
    else:
        # Try ML mask API first (produces clean wall-only mask)
        binary_mask = _try_ml_mask(img_bytes, image.filename or "upload.png")
        if binary_mask is not None:
            has_clean_mask = True
        else:
            # Fallback: adaptive threshold + cleanup pipeline
            gray = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY) if input_img.ndim == 3 else input_img
            binary_mask = _clean_threshold_mask(gray)

    # Use different defaults for clean ML masks vs noisy threshold masks.
    # ML masks: aggressive closing + uniform thickness for best quality.
    # Threshold masks: gentle closing, NO uniform thickness (would inflate
    # dimension lines/text to wall thickness making them indistinguishable).
    if has_clean_mask:
        eff_close = p.close_px
        eff_smooth = p.smooth_blur
        eff_ut = p.uniform_thickness
        eff_min_area = p.min_area
    else:
        # Raw floorplan → threshold mask cleaned by _clean_threshold_mask
        # (adaptive threshold + CC filter + skeleton normalization).
        # uniform_thickness=0: already normalized in cleanup step.
        # min_area=8000: filter furniture-sized holes (rooms > 10k px²).
        eff_close = p.close_px
        eff_smooth = p.smooth_blur
        eff_ut = 0                          # already normalized in _clean_threshold_mask
        eff_min_area = 8000                 # filter furniture outlines as holes

    # Create output subfolder: output/<timestamp>/
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = ROOT / "output" / ts
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save input image
    cv2.imwrite(str(out_dir / "input.png"), input_img)

    # Save cleaned binary mask (skeleton)
    cv2.imwrite(str(out_dir / "skeleton.png"), binary_mask)

    # Write mask for extract_wall_shapes (reads from path)
    mask_path = str(out_dir / "wall_mask.png")
    cv2.imwrite(mask_path, binary_mask)

    try:
        graph = extract_wall_shapes(
            mask_path,
            epsilon=p.epsilon,
            min_area=eff_min_area,
            close_px=eff_close,
            smooth_blur=eff_smooth,
            dilate_px=p.dilate_px,
            snap_axes=p.snap_axes,
            snap_tol_deg=p.snap_tol_deg,
            uniform_thickness=eff_ut,
            min_extent=p.min_extent,
            notch_depth=p.notch_depth,
            detect_openings=False,
            door_mask_path=None,
            window_mask_path=None,
        )
    except Exception:
        raise

    # Save graph.json
    with open(out_dir / "graph.json", "w") as f:
        json.dump(graph, f, indent=2)

    elapsed = time.perf_counter() - t0
    n_shapes = len(graph["shapes"])
    n_holes = sum(len(s["holes"]) for s in graph["shapes"])

    log.info("Output saved to %s (%.3fs)", out_dir, elapsed)

    return JSONResponse(
        content=graph,
        headers={
            "X-Shapes": str(n_shapes),
            "X-Rooms": str(n_holes),
            "X-Processing-Time": f"{elapsed:.3f}s",
            "X-Output-Dir": ts,
        },
    )


@app.post("/merge/openings", tags=["pipeline"])
def merge_openings(req: MergeRequest) -> JSONResponse:
    """
    Inject LLM-detected doors/windows into a wall graph.

    The caller provides:
    - `graph`: output from POST /extract/walls
    - `openings`: list of Opening objects detected by the external LLM

    Each Opening must specify:
    - `kind`: "door" or "window"
    - `x`, `y`: center position in original image pixels
    - `width`: opening span along the wall (pixels)
    - `wall_axis`: "h" (horizontal wall) or "v" (vertical wall)
    - `depth`: wall thickness estimate (pixels), default 10

    Returns the merged JSON ready for the THREE.js viewer.
    """
    graph = dict(req.graph)

    # Validate graph structure
    if graph.get("type") != "shape" or "shapes" not in graph:
        raise HTTPException(
            status_code=422,
            detail="'graph' must be a 'shape' JSON from POST /extract/walls.",
        )

    doors: list[dict] = []
    windows: list[dict] = []
    d_idx = 0
    w_idx = 0

    for op in req.openings:
        entry = {
            "id": op.id or (f"D{d_idx:04d}" if op.kind == "door" else f"W{w_idx:04d}"),
            "x": op.x,
            "y": op.y,
            "width": op.width,
            "depth": op.depth,
            "wall_axis": op.wall_axis,
        }
        if op.kind == "door":
            entry["door_type"] = "swing"
            doors.append(entry)
            if not op.id:
                d_idx += 1
        else:
            windows.append(entry)
            if not op.id:
                w_idx += 1

    graph["doors"] = doors
    graph["windows"] = windows

    return JSONResponse(content=graph)


# ── Entry point ────────────────────────────────────────────────────────────────


def main() -> None:
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8080,
        reload=False,
        log_level="info",
    )


if __name__ == "__main__":
    main()
