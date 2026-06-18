#!/usr/bin/env python3
"""Run wallgraph pipeline on 17817183169a88.png with full debug output."""

import cv2
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from wallgraph import WallDetector, WallGraphBuilder, WallGraphValidator

IMAGE = "sample/raw/17817183169a88.png"
OUT_PREFIX = "debug_178"

# --- Step 1: Wall detection (no snap_distance param — that's for builder) ---
detector = WallDetector(params={"min_wall_length": 40})

# Load image manually so we can capture intermediate masks
img = cv2.imread(IMAGE)
if img is None:
    print(f"ERROR: Cannot load image {IMAGE}", file=sys.stderr)
    sys.exit(1)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
print(f"Image loaded: {IMAGE}  shape={img.shape}")

# ---- Reproduce key steps of _detect_binary for debug masks ----

# Adaptive threshold
h, w_img = gray.shape
block_size = min(51, max(11, w_img // 10 | 1))
binary = cv2.adaptiveThreshold(
    gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY_INV, block_size, 10,
)
cv2.imwrite(f"{OUT_PREFIX}_binary_thresh.png", binary)
print(f"Saved: {OUT_PREFIX}_binary_thresh.png")

# Morph close
k_close = cv2.getStructuringElement(cv2.MORPH_RECT, (detector.params["close_kernel"],) * 2)
closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k_close, iterations=detector.params["close_iterations"])
cv2.imwrite(f"{OUT_PREFIX}_morph_close.png", closed)
print(f"Saved: {OUT_PREFIX}_morph_close.png")

# Morph open
k_open = cv2.getStructuringElement(cv2.MORPH_RECT, (detector.params["open_kernel"],) * 2)
opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, k_open)
cv2.imwrite(f"{OUT_PREFIX}_morph_open.png", opened)
print(f"Saved: {OUT_PREFIX}_morph_open.png")

# Remove small components (cleaned mask)
cleaned = detector._remove_small_components(opened)
cv2.imwrite(f"{OUT_PREFIX}_cleaned_mask.png", cleaned)
print(f"Saved: {OUT_PREFIX}_cleaned_mask.png")

# Thickness filter (thick mask for arcs)
thick_mask = detector._thickness_filter(cleaned)
thick_cleaned = detector._remove_small_components(thick_mask)
cv2.imwrite(f"{OUT_PREFIX}_thick_mask.png", thick_cleaned)
print(f"Saved: {OUT_PREFIX}_thick_mask.png")

# ---- Now run the full detect() pipeline ----
walls = detector.detect(IMAGE)
print(f"\nRaw walls detected: {len(walls)}")
for w in walls[:10]:
    print(f"  {w.id}: ({w.start.x:.1f},{w.start.y:.1f}) -> ({w.end.x:.1f},{w.end.y:.1f})  length={w.length():.1f}")
if len(walls) > 10:
    print(f"  ... and {len(walls)-10} more")

# ---- Step 2: Build wall graph ----
builder = WallGraphBuilder(snap_distance=5.0, min_wall_length=40)
graph = builder.build(walls)
# Count unique endpoints as "nodes"
endpoints = set()
for gw in graph.walls:
    endpoints.add((round(gw.start.x, 1), round(gw.start.y, 1)))
    endpoints.add((round(gw.end.x, 1), round(gw.end.y, 1)))
print(f"\nWall graph built: {len(graph.walls)} walls, {len(endpoints)} unique endpoints")

# ---- Step 3: Validate ----
validator = WallGraphValidator(snap_distance=5.0)
report = validator.validate(graph)
print(f"\nValidation: passed={report.passed}")
if report.errors:
    print(f"  Errors ({len(report.errors)}):")
    for e in report.errors[:5]:
        print(f"    {e}")
    if len(report.errors) > 5:
        print(f"    ... and {len(report.errors)-5} more")
if report.warnings:
    print(f"  Warnings ({len(report.warnings)}):")
    for w in report.warnings[:5]:
        print(f"    {w}")
    if len(report.warnings) > 5:
        print(f"    ... and {len(report.warnings)-5} more")

# ---- Step 4: Final visualization ----
detector.visualize(IMAGE, graph.walls, f"{OUT_PREFIX}_result.png")
print(f"\nSaved: {OUT_PREFIX}_result.png")

# ---- Step 5: Overlay walls on cleaned mask ----
overlay = cv2.cvtColor(cleaned, cv2.COLOR_GRAY2BGR)
for gw in graph.walls:
    color = (0, 255, 0)  # green for straight walls
    cv2.line(overlay,
             (int(gw.start.x), int(gw.start.y)),
             (int(gw.end.x), int(gw.end.y)),
             color, 2)
cv2.imwrite(f"{OUT_PREFIX}_overlay_cleaned.png", overlay)
print(f"Saved: {OUT_PREFIX}_overlay_cleaned.png")

# ---- Step 6: Overlay walls on original image ----
orig_overlay = img.copy()
for gw in graph.walls:
    cv2.line(orig_overlay,
             (int(gw.start.x), int(gw.start.y)),
             (int(gw.end.x), int(gw.end.y)),
             (0, 255, 0), 2)
    # Start point = blue, end point = red
    cv2.circle(orig_overlay, (int(gw.start.x), int(gw.start.y)), 4, (255, 0, 0), -1)
    cv2.circle(orig_overlay, (int(gw.end.x), int(gw.end.y)), 4, (0, 0, 255), -1)
cv2.imwrite(f"{OUT_PREFIX}_overlay_original.png", orig_overlay)
print(f"Saved: {OUT_PREFIX}_overlay_original.png")

print("\n=== SUMMARY ===")
print(f"Raw walls:     {len(walls)}")
print(f"Graph walls:   {len(graph.walls)}")
print(f"Endpoints:    {len(endpoints)}")
print(f"Validation:    passed={report.passed}, errors={len(report.errors)}, warnings={len(report.warnings)}")
