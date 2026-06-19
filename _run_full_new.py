#!/usr/bin/env python3
"""Run full wall detection pipeline with new _build_wall_mask and compare results."""

import cv2, numpy as np, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wallgraph import WallDetector, WallGraphBuilder, WallGraphValidator

IMAGE = "sample/raw/17817183169a88.png"
OUT = "sample/episodes/episode-8"

# Run with new classification
detector = WallDetector()
walls = detector.detect(IMAGE)
print(f"Walls detected: {len(walls)}")

builder = WallGraphBuilder(snap_distance=5.0, min_wall_length=40)
graph = builder.build(walls)
print(f"Graph walls: {len(graph.walls)}")

validator = WallGraphValidator(snap_distance=5.0)
report = validator.validate(graph)
print(f"Validation: passed={report.passed}")

# Visualize
detector.visualize(IMAGE, graph.walls, f"{OUT}/debug_178_result_new.png")

# Overlay walls on original
img = cv2.imread(IMAGE)
overlay = img.copy()
for gw in graph.walls:
    cv2.line(overlay,
             (int(gw.start.x), int(gw.start.y)),
             (int(gw.end.x), int(gw.end.y)),
             (0, 255, 0), 2)
    cv2.circle(overlay, (int(gw.start.x), int(gw.start.y)), 4, (255, 0, 0), -1)
    cv2.circle(overlay, (int(gw.end.x), int(gw.end.y)), 4, (0, 0, 255), -1)
cv2.imwrite(f"{OUT}/debug_178_overlay_new.png", overlay)

# Also produce the component wall mask
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
h, w_img = gray.shape
block_size = min(51, max(11, w_img // 10 | 1))
binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY_INV, block_size, 10)
k_close = cv2.getStructuringElement(cv2.MORPH_RECT, (detector.params["close_kernel"],) * 2)
closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k_close,
                           iterations=detector.params["close_iterations"])
k_open = cv2.getStructuringElement(cv2.MORPH_RECT, (detector.params["open_kernel"],) * 2)
opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, k_open)
cleaned = detector._remove_small_components(opened)
wall_mask = detector._build_wall_mask(cleaned)
cv2.imwrite(f"{OUT}/debug_component_wall_mask_final.png", wall_mask)

print(f"\nSaved: {OUT}/debug_178_result_new.png")
print(f"Saved: {OUT}/debug_178_overlay_new.png")
print(f"Saved: {OUT}/debug_component_wall_mask_final.png")
