"""
mask_to_shape.py — Convert binary wall mask to polygon shapes for Three.js ExtrudeGeometry.

Instead of skeleton → Hough line detection (which misses corners / creates gaps),
this module extracts exact wall contours via cv2.findContours and returns them as
2D polygon data. Three.js extrudes these polygons to create solid 3D walls.

Two pre-processing steps fix the common ML mask artifacts:
  1. MORPH_CLOSE  — fills small gaps/breaks at wall junctions and corners
  2. Gaussian blur + re-threshold — smooths staircase pixel edges on diagonals

Usage:
    from wallgraph.mask_to_shape import extract_wall_shapes
    data = extract_wall_shapes("output/wall_mask_image-3.png")
    # {"type": "shape", "shapes": [...], "image_size": [w, h]}
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np


def _preprocess_mask(
    binary: np.ndarray,
    close_px: int,
    smooth_blur: int,
    dilate_px: int,
) -> np.ndarray:
    """
    Clean binary mask before contour extraction.

    Steps (in order):
      1. MORPH_CLOSE  — dilate then erode to fill corner/junction gaps
      2. Gaussian blur + re-threshold — smooth staircase pixel edges
      3. Optional extra dilation — expand wall width slightly for micro-gap fill
    """
    result = binary.copy()

    # 1. Morphological closing: closes breaks at wall junctions/corners.
    #    Kernel size close_px*2+1 fills gaps up to close_px pixels wide.
    if close_px > 0:
        k = close_px * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel, iterations=2)

    # 2. Gaussian blur + re-threshold: anti-aliases the jagged pixel boundary.
    #    Blur spreads the edge over several pixels → after thresholding at 127
    #    the boundary falls at the smooth midpoint → far fewer staircase steps.
    if smooth_blur > 0:
        blur_k = smooth_blur * 2 + 1  # must be odd
        blurred = cv2.GaussianBlur(result, (blur_k, blur_k), 0)
        _, result = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)

    # 3. Optional extra dilation (dilate_px) — additional wall-width expansion.
    if dilate_px > 0:
        k = dilate_px * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
        result = cv2.dilate(result, kernel, iterations=1)

    return result


def _collapse_staircases(
    pts: list[list[float]],
    max_step_px: float = 8.0,
) -> list[list[float]]:
    """
    Collapse staircase H-V-H or V-H-V patterns into straight segments.

    After H/V snapping, ML mask noise produces alternating H-V edges where
    the minor-direction displacement is tiny (staircase effect). This function
    detects those patterns and collapses them:

      H→small_V→H  (step < max_step_px) → single H at median Y
      V→small_H→V  (step < max_step_px) → single V at median X

    Multiple passes iterate until no more collapses are possible.
    """
    for _ in range(16):
        n = len(pts)
        if n < 4:
            break

        # Classify each edge as 'h' or 'v'
        etypes: list[str] = []
        for i in range(n):
            j = (i + 1) % n
            dx = abs(pts[j][0] - pts[i][0])
            dy = abs(pts[j][1] - pts[i][1])
            etypes.append('h' if dx >= dy else 'v')

        keep = [True] * n
        changed = False

        for i in range(n):
            if not keep[i]:
                continue
            i1 = (i + 1) % n
            i2 = (i + 2) % n
            if not keep[i1]:
                continue

            p0, p1, p2 = pts[i], pts[i1], pts[i2]
            e0, e1 = etypes[i], etypes[i1]

            # H → small_V → next_H: collapse p1, snap p2.y = p0.y
            if e0 == 'h' and e1 == 'v' and etypes[i2] == 'h':
                if abs(p2[1] - p1[1]) < max_step_px:
                    keep[i1] = False
                    pts[i2] = [pts[i2][0], p0[1]]
                    changed = True
                    continue

            # V → small_H → next_V: collapse p1, snap p2.x = p0.x
            if e0 == 'v' and e1 == 'h' and etypes[i2] == 'v':
                if abs(p2[0] - p1[0]) < max_step_px:
                    keep[i1] = False
                    pts[i2] = [p0[0], pts[i2][1]]
                    changed = True
                    continue

        pts = [p for p, k in zip(pts, keep) if k]
        if not changed:
            break

    return pts if len(pts) >= 3 else pts


def _merge_coaxial(pts: list[list[float]], tol: float = 1.0) -> list[list[float]]:
    """
    Remove redundant coaxial vertices after H/V snapping.

    If three consecutive vertices A→B→C lie on the same horizontal or vertical
    line (A.y == B.y == C.y, or A.x == B.x == C.x), then B is redundant and
    is removed. Multiple passes collapse long collinear chains.
    """
    for _ in range(8):
        n = len(pts)
        if n < 4:
            break
        new_pts: list[list[float]] = []
        changed = False
        for i in range(n):
            prev = pts[(i - 1) % n]
            curr = pts[i]
            nxt  = pts[(i + 1) % n]
            same_y = abs(prev[1] - curr[1]) < tol and abs(curr[1] - nxt[1]) < tol
            same_x = abs(prev[0] - curr[0]) < tol and abs(curr[0] - nxt[0]) < tol
            if same_y or same_x:
                changed = True  # collinear, remove curr
            else:
                new_pts.append(curr)
        pts = new_pts
        if not changed:
            break
    return pts if len(pts) >= 3 else pts


def _remove_hv_notches(
    pts: list[list[float]],
    max_depth: float = 15.0,
    passes: int = 8,
) -> list[list[float]]:
    """
    Remove rectangular notches (window bumps) from H/V-snapped polygons.

    After _snap_to_axis all edges are either horizontal or vertical.
    A notch is a 4-vertex pattern where the polygon deviates perpendicularly
    and RETURNS to the same axis:
        H → (short V up/down) → H back at same Y   (horizontal notch)
        V → (short H left/right) → V back at same X (vertical notch)

    This is safe for room corners because:
        Real corner: H → V → stays V (does NOT return)  → NEVER removed
        Window notch: H → V → H back   (returns)        → REMOVED if depth < max_depth

    The depth is the perpendicular excursion (absolute Y or X offset).
    """
    for _ in range(passes):
        n = len(pts)
        if n < 5:
            break
        keep = [True] * n
        changed = False
        for i in range(n):
            if not keep[i]:
                continue
            # 4-vertex window: p0=i, p1=i+1, p2=i+2, p3=i+3
            i1 = (i + 1) % n
            i2 = (i + 2) % n
            i3 = (i + 3) % n
            if not (keep[i1] and keep[i2] and keep[i3]):
                continue
            p0, p1, p2, p3 = pts[i], pts[i1], pts[i2], pts[i3]

            # Horizontal notch: p0 and p3 share same Y
            if abs(p0[1] - p3[1]) < 1.0:
                depth = max(abs(p1[1] - p0[1]), abs(p2[1] - p0[1]))
                if depth < max_depth:
                    keep[i1] = keep[i2] = False
                    changed = True
                    continue

            # Vertical notch: p0 and p3 share same X
            if abs(p0[0] - p3[0]) < 1.0:
                depth = max(abs(p1[0] - p0[0]), abs(p2[0] - p0[0]))
                if depth < max_depth:
                    keep[i1] = keep[i2] = False
                    changed = True

        pts = [p for p, k in zip(pts, keep) if k]
        if not changed:
            break
    return pts if len(pts) >= 3 else pts


def _straighten_hv_runs(
    pts: list[list[float]],
    angle_tol_deg: float = 25.0,
) -> list[list[float]]:
    """
    Straighten consecutive near-H/V edges into perfectly straight wall runs.

    Unlike per-edge snapping (_snap_to_axis), this groups consecutive edges
    that all point in the same approximate direction (H or V) into a "wall
    run". All vertices in a run are then set to the SAME median coordinate
    (median Y for H runs, median X for V runs). This avoids the vertex-conflict
    issue of per-edge snapping where the same vertex is modified by two
    different adjacent edge snaps, creating diagonal artifacts.

    Real 45° edges and junctions (short diagonal transition vertices between
    H and V runs) are left untouched — they naturally snap to the adjacent
    run's coordinate via _merge_coaxial afterward.

    angle_tol_deg : classify edge as H if within this many degrees of
                    horizontal, V if within this many degrees of vertical.
                    Default 25° handles ML mask noise well. True 45° diagonals
                    (> 45° from both axes) are left as-is.
    """
    n = len(pts)
    if n < 3:
        return pts

    tol = np.tan(np.radians(angle_tol_deg))

    def edge_type(i: int, p: list[list[float]]) -> str:
        j = (i + 1) % len(p)
        dx = p[j][0] - p[i][0]
        dy = p[j][1] - p[i][1]
        length = np.hypot(dx, dy)
        if length < 1e-6:
            return "h"
        if abs(dy / length) < tol:
            return "h"
        if abs(dx / length) < tol:
            return "v"
        return "d"

    # Compute all edge types once
    types = [edge_type(i, pts) for i in range(n)]

    # Find index of first diagonal (or 0) to start run detection cleanly
    start = 0
    for k in range(n):
        if types[k] == "d":
            start = (k + 1) % n
            break

    out = [list(p) for p in pts]
    visited = [False] * n

    for offset in range(n):
        ei = (start + offset) % n
        if visited[ei]:
            continue
        if types[ei] == "d":
            visited[ei] = True
            continue

        # Collect maximal run of same axis type
        rtype = types[ei]
        run: list[int] = []
        k = ei
        while not visited[k] and types[k] == rtype:
            run.append(k)
            visited[k] = True
            k = (k + 1) % n
            if k == ei:
                break  # full circle (all-H or all-V polygon)

        # Gather unique vertex indices in this run
        seen: set[int] = set()
        verts: list[int] = []
        for e in run:
            for v in (e, (e + 1) % n):
                if v not in seen:
                    seen.add(v)
                    verts.append(v)

        # Compute median coordinate and apply
        if rtype == "h":
            coords = [out[v][1] for v in verts]
            med = sorted(coords)[len(coords) // 2]
            for v in verts:
                out[v][1] = med
        else:
            coords = [out[v][0] for v in verts]
            med = sorted(coords)[len(coords) // 2]
            for v in verts:
                out[v][0] = med

    return out


def _skeletonize(binary: np.ndarray) -> np.ndarray:
    """1-pixel skeleton via ximgproc.thinning or iterative morphological fallback."""
    img = (binary > 0).astype(np.uint8) * 255
    try:
        return cv2.ximgproc.thinning(img, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)  # type: ignore[attr-defined]
    except AttributeError:
        skel = np.zeros_like(img)
        element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        temp = img.copy()
        while True:
            eroded = cv2.erode(temp, element)
            opened = cv2.dilate(eroded, element)
            subset = cv2.subtract(temp, opened)
            skel = cv2.bitwise_or(skel, subset)
            temp = eroded.copy()
            if cv2.countNonZero(temp) == 0:
                break
        return skel


def _normalize_thickness(binary: np.ndarray, half_px: int) -> np.ndarray:
    """
    Normalize wall thickness to uniform width.

    1. Skeletonize → 1px centerlines (strips uneven ML mask widths)
    2. Re-dilate by half_px → all walls become 2*half_px px thick
    """
    skel = _skeletonize(binary)
    k = half_px * 2 + 1
    # MORPH_RECT: sharp rectangular corners at wall junctions/room corners.
    # MORPH_ELLIPSE was rounding corners → diagonal artifacts in small rooms.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
    return cv2.dilate(skel, kernel, iterations=1)


def extract_wall_shapes(
    mask_path: str,
    epsilon: float = 4.0,
    min_area: int = 300,
    close_px: int = 5,
    smooth_blur: int = 3,
    dilate_px: int = 0,
    snap_axes: bool = True,
    snap_tol_deg: float = 8.0,
    uniform_thickness: int = 0,
    min_extent: int = 0,
    notch_depth: float = 0.0,
    # Opening detection
    detect_openings: bool = False,
    door_mask_path: str | None = None,
    window_mask_path: str | None = None,
) -> dict:
    """
    Extract wall polygon shapes from a binary wall mask.

    Parameters
    ----------
    mask_path : str
        Path to binary mask PNG (white=wall, black=empty).
    epsilon : float
        Douglas-Peucker polygon simplification (px).
        Larger = fewer vertices = straighter walls. Default 4.0.
        Try 6-10 for very clean output; 2-3 for maximum accuracy.
    min_area : int
        Minimum contour area (px²) — filters noise / tiny artifacts.
    close_px : int
        Morphological closing kernel radius (px). Closes gaps at wall
        corners and T-junctions. Default 5. Set 0 to disable.
    smooth_blur : int
        Gaussian blur radius for edge smoothing before contour extraction.
        Eliminates staircase pixel steps on diagonal edges. Default 3.
        Set 0 to disable.
    dilate_px : int
        Extra dilation after closing/blur (px). Default 0 (disabled).
    snap_axes : bool
        Snap near-H/V edges to exact horizontal/vertical. Default True.
    snap_tol_deg : float
        Angle tolerance for axis snapping (degrees). Default 8.
    uniform_thickness : int
        If > 0, normalize all walls to this half-thickness (px) by:
        skeletonizing the mask → re-dilating by uniform_thickness.
        Fixes uneven ML mask wall widths. Default 0 (disabled).
        Typical value: 8 (→ 16px = ~15cm walls at 1px≈1cm scale).
    min_extent : int
        Minimum size of the longer side of the contour bounding box (px).
        Filters small blobby noise (door stubs, furniture fragments) while
        keeping long thin walls. Default 0 (disabled).
        Typical value: 40-60px to remove small isolated fragments.
    notch_depth : float
        Max perpendicular excursion (px) for H/V notch removal. Removes
        rectangular bump patterns (H→V→H or V→H→V that RETURN to same axis)
        where deviation < notch_depth. Safe: real 90° corners never return
        to same axis so are never removed. Default 0 (disabled).
        Typical: 10-20px for window notch / mask roughness cleanup.

    detect_openings : bool
        If True, run geometric door/window detection from wall mask gaps.
        Uses the original mask (before closing) to find openings.
    door_mask_path : str | None
        Path to dedicated door mask PNG from ML API (overrides geometric detection).
    window_mask_path : str | None
        Path to dedicated window mask PNG from ML API (overrides geometric detection).

    Returns
    -------
    dict with keys:
        type       : "shape"
        shapes     : list of {outer: [[x,y],...], holes: [[[x,y],...],...]}
        doors      : list of door openings (present if detect_openings or door_mask_path)
        windows    : list of window openings (present if detect_openings or window_mask_path)
        image_size : [width, height]
    """
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Cannot load mask: {mask_path}")

    _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    h, w = mask.shape

    # Pre-process: close gaps, smooth edges
    binary = _preprocess_mask(binary, close_px=close_px, smooth_blur=smooth_blur, dilate_px=dilate_px)

    # Optional: normalize wall thickness (skeleton → uniform re-dilation)
    if uniform_thickness > 0:
        binary = _normalize_thickness(binary, half_px=uniform_thickness)

    # RETR_CCOMP: 2-level hierarchy
    #   Level 0: outer boundary of white (wall) regions
    #   Level 1: inner holes (black room interiors)
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    if hierarchy is None or len(contours) == 0:
        return {"type": "shape", "shapes": [], "image_size": [w, h]}

    hier = hierarchy[0]  # shape (N, 4): [next_sib, prev_sib, first_child, parent]
    shapes: list[dict] = []

    def simplify(cnt: np.ndarray) -> list[list[float]]:
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        pts: list[list[float]] = approx.reshape(-1, 2).tolist()
        if snap_axes and len(pts) >= 3:
            for _ in range(3):  # 3 passes: each pass collapses junction artifacts created by the previous
                pts = _straighten_hv_runs(pts, angle_tol_deg=snap_tol_deg)
                if len(pts) >= 4:
                    pts = _merge_coaxial(pts)
                if len(pts) < 3:
                    break
            # Collapse staircase H-V-H / V-H-V noise patterns
            if len(pts) >= 4:
                pts = _collapse_staircases(pts, max_step_px=8.0)
                pts = _merge_coaxial(pts)
            # Remove rectangular notches (window bumps, mask roughness)
            if notch_depth > 0 and len(pts) >= 5:
                pts = _remove_hv_notches(pts, max_depth=notch_depth)
                pts = _merge_coaxial(pts)
        return pts

    for i, cnt in enumerate(contours):
        # Only root contours (no parent → outer wall boundary)
        if hier[i][3] != -1:
            continue

        area = cv2.contourArea(cnt)
        if area < min_area:
            continue

        # min_extent: filter by bounding box longer side
        # Catches small blobby fragments (door stubs, furniture noise)
        # without removing long thin walls that have low area
        if min_extent > 0:
            _, _, bw, bh = cv2.boundingRect(cnt)
            if max(bw, bh) < min_extent:
                continue

        outer = simplify(cnt)
        if len(outer) < 3:
            continue

        # Collect room holes (direct children)
        holes: list[list[list[float]]] = []
        child_idx = hier[i][2]
        while child_idx != -1:
            child_cnt = contours[child_idx]
            if cv2.contourArea(child_cnt) >= min_area:
                hole_pts = simplify(child_cnt)
                if len(hole_pts) >= 3:
                    holes.append(hole_pts)
            child_idx = hier[child_idx][0]  # next sibling

        shapes.append({"outer": outer, "holes": holes})

    result: dict = {"type": "shape", "shapes": shapes, "image_size": [w, h]}

    # ── Opening detection ──────────────────────────────────────────────────
    if door_mask_path or window_mask_path:
        from .opening_detector import detect_from_ml_masks
        doors, windows = detect_from_ml_masks(door_mask_path, window_mask_path)
        result["doors"]   = doors
        result["windows"] = windows
    elif detect_openings:
        from .opening_detector import detect_from_gap_scan
        doors, windows = detect_from_gap_scan(mask_path)
        result["doors"]   = doors
        result["windows"] = windows

    return result


def save_shapes(data: dict, out_path: str) -> None:
    """Save shape data to JSON file."""
    Path(out_path).write_text(json.dumps(data, indent=2))


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m wallgraph.mask_to_shape <mask.png> [out.json]")
        sys.exit(1)

    mask_path = sys.argv[1]
    out_path  = sys.argv[2] if len(sys.argv) > 2 else None

    data = extract_wall_shapes(mask_path)
    n_shapes = len(data["shapes"])
    n_holes  = sum(len(s["holes"]) for s in data["shapes"])
    print(f"Extracted {n_shapes} wall shape(s), {n_holes} room hole(s)")
    print(f"Image size: {data['image_size']}")

    if out_path:
        save_shapes(data, out_path)
        print(f"Saved → {out_path}")
    else:
        print(json.dumps(data, indent=2))
