"""
visualizer.py — Debug visualization for wall shapes and connector points.

Generates two test images from extract_wall_shapes() JSON output:

  skeleton.png  — Wall polygon outlines on white canvas.
                  Shows exact shape geometry (outer + holes).
                  Good for verifying wall straightness and polygon integrity.

  connector.png — Same + all polygon vertices marked as colored dots.
                  Shows connection points (corners, T-junctions, L-junctions).
                  Good for verifying vertex count and junction quality.

Optional overlays:
  - Doors   (cyan rectangle at gap position)
  - Windows (yellow rectangle at gap position)
  - Holes   (semi-transparent fill to distinguish room interiors)

Usage:
    from wallgraph.visualizer import render_skeleton, render_connector

    data = extract_wall_shapes("output/wall_mask_image-4.png", ...)
    render_skeleton("output/skeleton_image-4.png", data)
    render_connector("output/connector_image-4.png", data)
"""

from __future__ import annotations

import math
import cv2
import numpy as np


# ── Colors (BGR) ─────────────────────────────────────────────────────────────

C_BG         = (255, 255, 255)   # white background
C_WALL       = (40,  40,  40)    # dark gray wall fill
C_WALL_EDGE  = (0,   0,   0)     # black wall outline
C_HOLE       = (220, 230, 240)   # light blue room interior
C_HOLE_EDGE  = (100, 130, 160)   # blue-gray room border
C_VERTEX     = (0,   0,   230)   # blue: outer polygon vertices
C_HOLE_VERT  = (200, 100, 0)     # orange: hole vertices
C_DOOR       = (0,   200, 200)   # cyan: door opening
C_WINDOW     = (0,   200, 80)    # green: window opening
C_LABEL      = (60,  60,  180)   # label text


def _pts_to_np(pts: list[list[float]]) -> np.ndarray:
    return np.array(pts, dtype=np.int32).reshape(-1, 1, 2)


def render_skeleton(
    out_path: str,
    data: dict,
    scale: float = 1.0,
    show_holes: bool = True,
    show_doors: bool = True,
    show_windows: bool = True,
) -> np.ndarray:
    """
    Draw wall polygon outlines on white canvas.

    Parameters
    ----------
    out_path    : output PNG path
    data        : dict from extract_wall_shapes()
    scale       : scale factor (1.0 = original size)
    show_holes  : shade room interiors
    show_doors  : draw door opening rectangles
    show_windows: draw window opening rectangles

    Returns
    -------
    BGR image as numpy array
    """
    w, h = data["image_size"]
    sw, sh = int(w * scale), int(h * scale)

    canvas = np.full((sh, sw, 3), C_BG, dtype=np.uint8)

    for shape in data["shapes"]:
        outer = _pts_to_np([[p[0] * scale, p[1] * scale] for p in shape["outer"]])

        # Fill wall body
        cv2.fillPoly(canvas, [outer], C_WALL)

        # Shade room holes (interiors)
        for hole in shape.get("holes", []):
            hole_pts = _pts_to_np([[p[0] * scale, p[1] * scale] for p in hole])
            if show_holes:
                cv2.fillPoly(canvas, [hole_pts], C_HOLE)
            cv2.polylines(canvas, [hole_pts], True, C_HOLE_EDGE, 1)

        # Outer wall outline
        cv2.polylines(canvas, [outer], True, C_WALL_EDGE, 1)

    # Draw doors
    if show_doors:
        for door in data.get("doors", []):
            _draw_opening(canvas, door, C_DOOR, scale, label=door.get("id", "D?"))

    # Draw windows
    if show_windows:
        for win in data.get("windows", []):
            _draw_opening(canvas, win, C_WINDOW, scale, label=win.get("id", "W?"))

    cv2.imwrite(out_path, canvas)
    return canvas


def render_connector(
    out_path: str,
    data: dict,
    scale: float = 1.0,
    vertex_radius: int = 4,
    show_doors: bool = True,
    show_windows: bool = True,
) -> np.ndarray:
    """
    Draw wall polygons + all vertices as colored dots.

    Outer polygon vertices: blue dots
    Hole (room) vertices:   orange dots

    A high vertex count at a junction indicates multiple redundant points —
    this helps tune the epsilon / snap parameters.

    Parameters
    ----------
    out_path      : output PNG path
    data          : dict from extract_wall_shapes()
    scale         : scale factor
    vertex_radius : radius of vertex dots (px)

    Returns
    -------
    BGR image as numpy array
    """
    # Build base canvas without saving (reuse skeleton render logic)
    w, h = data["image_size"]
    sw, sh = int(w * scale), int(h * scale)
    canvas = np.full((sh, sw, 3), C_BG, dtype=np.uint8)

    for shape in data["shapes"]:
        outer = _pts_to_np([[p[0] * scale, p[1] * scale] for p in shape["outer"]])
        cv2.fillPoly(canvas, [outer], C_WALL)
        for hole in shape.get("holes", []):
            hole_pts = _pts_to_np([[p[0] * scale, p[1] * scale] for p in hole])
            cv2.fillPoly(canvas, [hole_pts], C_HOLE)
            cv2.polylines(canvas, [hole_pts], True, C_HOLE_EDGE, 1)
        cv2.polylines(canvas, [outer], True, C_WALL_EDGE, 1)

    if show_doors:
        for door in data.get("doors", []):
            _draw_opening(canvas, door, C_DOOR, scale, label=door.get("id", "D?"))
    if show_windows:
        for win in data.get("windows", []):
            _draw_opening(canvas, win, C_WINDOW, scale, label=win.get("id", "W?"))

    outer_count = 0
    hole_count  = 0

    for shape in data["shapes"]:
        # Outer polygon vertices
        for pt in shape["outer"]:
            cx, cy = int(pt[0] * scale), int(pt[1] * scale)
            cv2.circle(canvas, (cx, cy), vertex_radius, C_VERTEX, -1)
            cv2.circle(canvas, (cx, cy), vertex_radius + 1, (0, 0, 0), 1)
            outer_count += 1

        # Hole vertices
        for hole in shape.get("holes", []):
            for pt in hole:
                cx, cy = int(pt[0] * scale), int(pt[1] * scale)
                cv2.circle(canvas, (cx, cy), vertex_radius - 1, C_HOLE_VERT, -1)
                cv2.circle(canvas, (cx, cy), vertex_radius, (0, 0, 0), 1)
                hole_count += 1

    # Stats overlay
    n_shapes  = len(data["shapes"])
    n_holes   = sum(len(s.get("holes", [])) for s in data["shapes"])
    n_doors   = len(data.get("doors", []))
    n_windows = len(data.get("windows", []))

    lines = [
        f"shapes: {n_shapes}  holes: {n_holes}",
        f"outer verts: {outer_count}  hole verts: {hole_count}",
        f"doors: {n_doors}  windows: {n_windows}",
    ]
    _draw_legend(canvas, lines)

    cv2.imwrite(out_path, canvas)
    return canvas


def render_both(
    stem: str,
    data: dict,
    scale: float = 1.0,
) -> tuple[str, str]:
    """
    Convenience: render both skeleton and connector images.

    Parameters
    ----------
    stem : file path stem, e.g. "output/debug_image-4"
           Generates: stem + "_skeleton.png" and stem + "_connector.png"

    Returns
    -------
    (skeleton_path, connector_path)
    """
    skel_path = f"{stem}_skeleton.png"
    conn_path = f"{stem}_connector.png"
    render_skeleton(skel_path, data, scale=scale)
    render_connector(conn_path, data, scale=scale)
    return skel_path, conn_path


# ── Helpers ───────────────────────────────────────────────────────────────────

def _draw_opening(
    canvas: np.ndarray,
    opening: dict,
    color: tuple[int, int, int],
    scale: float,
    label: str = "",
) -> None:
    """Draw a door or window as a filled rectangle at its gap position."""
    cx   = int(opening["x"]     * scale)
    cy   = int(opening["y"]     * scale)
    half_w = int(opening["width"] * scale / 2)
    half_d = max(2, int(opening.get("depth", 10) * scale / 2))

    axis = opening.get("wall_axis", "h")
    if axis == "h":
        x1, y1 = cx - half_w, cy - half_d
        x2, y2 = cx + half_w, cy + half_d
    else:
        x1, y1 = cx - half_d, cy - half_w
        x2, y2 = cx + half_d, cy + half_w

    # Semi-transparent fill
    overlay = canvas.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
    cv2.addWeighted(overlay, 0.5, canvas, 0.5, 0, canvas)

    # Solid border
    cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)

    # Label
    if label:
        cv2.putText(
            canvas, label,
            (x1, y1 - 3),
            cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA,
        )


def _draw_legend(canvas: np.ndarray, lines: list[str]) -> None:
    """Draw stats box in top-left corner."""
    pad = 8
    lh  = 18
    box_h = pad * 2 + lh * len(lines)
    box_w = 300

    # Semi-transparent background
    overlay = canvas.copy()
    cv2.rectangle(overlay, (0, 0), (box_w, box_h), (240, 240, 240), -1)
    cv2.addWeighted(overlay, 0.75, canvas, 0.25, 0, canvas)
    cv2.rectangle(canvas, (0, 0), (box_w, box_h), (180, 180, 180), 1)

    for i, line in enumerate(lines):
        y = pad + lh * i + lh - 4
        cv2.putText(
            canvas, line, (pad, y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, C_LABEL, 1, cv2.LINE_AA,
        )
