"""
opening_detector.py — Detect doors and windows from floorplan masks.

Two detection modes:

1. Geometric gap scan (wall mask only):
   - Find black (empty) connected regions in wall mask
   - Check if sandwiched between wall pixels on both sides
   - Classify by bounding box size: door (20-120px) vs window (6-25px)
   - No extra ML needed, works with existing wall mask

2. ML masks (best accuracy):
   - Use dedicated door/window masks from extended CubiCasa API
   - Each mask: white = object, black = background
   - See docs/VPS_OPENINGS.md for how to extend the VPS API

Usage:
    from wallgraph.opening_detector import detect_from_gap_scan, detect_from_ml_masks

    # Geometric (wall mask only)
    doors, windows = detect_from_gap_scan("output/wall_mask.png")

    # ML masks (separate door + window masks from API)
    doors, windows = detect_from_ml_masks(
        door_mask_path="output/door_mask.png",
        window_mask_path="output/window_mask.png",
    )
"""

from __future__ import annotations

import cv2
import numpy as np

# ── Size thresholds (pixels) ──────────────────────────────────────────────────

DOOR_MIN_PX    = 20    # min opening width to classify as door
DOOR_MAX_PX    = 130   # max opening width to classify as door
WINDOW_MIN_PX  = 6     # min opening width to classify as window
WINDOW_MAX_PX  = 25    # max opening width (< DOOR_MIN_PX to avoid overlap)

# Min depth (perpendicular to wall): opening must go through most of the wall
OPENING_MIN_DEPTH = 3

# How far outside the gap bbox to look for confirming wall pixels
WALL_CONFIRM_PX = 12

# Min fraction of confirming strip that must be wall pixels
WALL_SIDE_MIN_DENSITY = 0.15

# Deduplication: openings closer than this are merged (pixels)
DEDUP_DIST = 18


# ── Geometric detection from wall mask gaps ───────────────────────────────────

def detect_from_gap_scan(
    wall_mask_path: str,
    door_min_px: int = DOOR_MIN_PX,
    door_max_px: int = DOOR_MAX_PX,
    window_min_px: int = WINDOW_MIN_PX,
    window_max_px: int = WINDOW_MAX_PX,
    light_close_px: int = 2,
) -> tuple[list[dict], list[dict]]:
    """
    Detect doors and windows by finding sandwiched gaps in wall mask.

    Reads the wall mask BEFORE the main morphological close (which fills
    door gaps). Uses only a tiny close (light_close_px) for noise cleanup.

    Parameters
    ----------
    wall_mask_path : path to binary wall mask PNG (white=wall)
    door_min_px / door_max_px : width range for door gaps (px)
    window_min_px / window_max_px : width range for window gaps (px)
    light_close_px : small close for noise only (not gap-filling)

    Returns
    -------
    (doors, windows) — lists of dicts ready for JSON output
        doors:   [{id, x, y, width, depth, wall_axis, door_type}, ...]
        windows: [{id, x, y, width, depth, wall_axis}, ...]
    """
    img = cv2.imread(wall_mask_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot load wall mask: {wall_mask_path}")

    _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

    # Tiny close: join pixel-level noise only, don't fill real openings
    if light_close_px > 0:
        k = light_close_px * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    return _detect_gaps(binary, door_min_px, door_max_px, window_min_px, window_max_px)


def _confirm_sandwiched(
    bbox: tuple[int, int, int, int],
    binary: np.ndarray,
    axis: str,
) -> bool:
    """Return True if the gap has wall pixels on BOTH perpendicular sides."""
    x, y, bw, bh = bbox
    h_img, w_img = binary.shape
    m = WALL_CONFIRM_PX

    if axis == "h":
        # Gap runs left-right → wall must be above AND below
        above = binary[max(0, y - m) : y,             x : x + bw]
        below = binary[y + bh : min(h_img, y + bh + m), x : x + bw]
        if above.size == 0 or below.size == 0:
            return False
        return (
            np.sum(above > 0) / above.size >= WALL_SIDE_MIN_DENSITY
            and np.sum(below > 0) / below.size >= WALL_SIDE_MIN_DENSITY
        )
    else:
        # Gap runs top-bottom → wall must be left AND right
        left  = binary[y : y + bh, max(0, x - m) : x            ]
        right = binary[y : y + bh, x + bw : min(w_img, x + bw + m)]
        if left.size == 0 or right.size == 0:
            return False
        return (
            np.sum(left > 0) / left.size >= WALL_SIDE_MIN_DENSITY
            and np.sum(right > 0) / right.size >= WALL_SIDE_MIN_DENSITY
        )


def _detect_gaps(
    binary: np.ndarray,
    door_min_px: int,
    door_max_px: int,
    window_min_px: int,
    window_max_px: int,
) -> tuple[list[dict], list[dict]]:
    """Find sandwiched empty regions in binary wall mask and classify as openings."""
    # Invert: empty (black) regions become white components
    inverted = cv2.bitwise_not(binary)

    num_labels, _labels, stats, centroids = cv2.connectedComponentsWithStats(inverted, 8)

    doors:   list[dict] = []
    windows: list[dict] = []
    seen_centers: list[tuple[float, float]] = []

    for i in range(1, num_labels):
        bx = int(stats[i, cv2.CC_STAT_LEFT])
        by = int(stats[i, cv2.CC_STAT_TOP])
        bw = int(stats[i, cv2.CC_STAT_WIDTH])
        bh = int(stats[i, cv2.CC_STAT_HEIGHT])

        # Determine opening axis from aspect ratio
        if bw >= bh * 1.5:
            axis     = "h"
            gap_size = bw
            gap_depth = bh
        elif bh >= bw * 1.5:
            axis     = "v"
            gap_size = bh
            gap_depth = bw
        else:
            # Near-square = room space, corner, or blob — not an opening
            continue

        if gap_depth < OPENING_MIN_DEPTH:
            continue

        is_door   = door_min_px   <= gap_size <= door_max_px
        is_window = window_min_px <= gap_size <= window_max_px

        if not is_door and not is_window:
            continue

        # Confirm wall on both sides of the gap
        if not _confirm_sandwiched((bx, by, bw, bh), binary, axis):
            continue

        cx = float(centroids[i][0])
        cy = float(centroids[i][1])

        # Deduplicate nearby detections
        duplicate = any(
            (cx - sx) ** 2 + (cy - sy) ** 2 < DEDUP_DIST ** 2
            for sx, sy in seen_centers
        )
        if duplicate:
            continue
        seen_centers.append((cx, cy))

        entry: dict = {
            "x":        round(cx, 1),
            "y":        round(cy, 1),
            "width":    float(gap_size),
            "depth":    float(gap_depth),
            "wall_axis": axis,
        }

        if is_door:
            entry["id"]        = f"D{len(doors):04d}"
            entry["door_type"] = "swing"
            doors.append(entry)
        else:
            entry["id"] = f"Win{len(windows):04d}"
            windows.append(entry)

    return doors, windows


# ── ML mask detection ─────────────────────────────────────────────────────────

def detect_from_ml_masks(
    door_mask_path: str | None = None,
    window_mask_path: str | None = None,
) -> tuple[list[dict], list[dict]]:
    """
    Detect doors and windows from dedicated ML segmentation masks.

    Designed for the extended CubiCasa API that returns separate masks.
    See docs/VPS_OPENINGS.md for how to add those endpoints to app.py.

    Parameters
    ----------
    door_mask_path : path to binary door mask PNG (white = door)
    window_mask_path : path to binary window mask PNG (white = window)

    Returns
    -------
    (doors, windows) as lists of dicts
    """
    doors:   list[dict] = []
    windows: list[dict] = []

    if door_mask_path:
        doors = _mask_to_openings(door_mask_path, prefix="D", kind="door")

    if window_mask_path:
        windows = _mask_to_openings(window_mask_path, prefix="Win", kind="window")

    return doors, windows


def _mask_to_openings(mask_path: str, prefix: str, kind: str) -> list[dict]:
    """Extract openings from a binary segmentation mask via connected components."""
    img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot load mask: {mask_path}")

    _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

    # Remove single-pixel noise
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    num_labels, _labels, stats, centroids = cv2.connectedComponentsWithStats(binary, 8)

    openings: list[dict] = []
    for i in range(1, num_labels):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < 50:
            continue

        bw = int(stats[i, cv2.CC_STAT_WIDTH])
        bh = int(stats[i, cv2.CC_STAT_HEIGHT])
        cx = float(centroids[i][0])
        cy = float(centroids[i][1])

        wall_axis = "h" if bw >= bh else "v"
        gap_size  = float(max(bw, bh))
        gap_depth = float(min(bw, bh))

        entry: dict = {
            "id":       f"{prefix}{len(openings):04d}",
            "x":        round(cx, 1),
            "y":        round(cy, 1),
            "width":    gap_size,
            "depth":    gap_depth,
            "wall_axis": wall_axis,
        }
        if kind == "door":
            entry["door_type"] = "swing"

        openings.append(entry)

    return openings
