"""Test script: run WallDetector on image-3.jpg and visualize results."""

import cv2
import numpy as np
import sys

sys.path.insert(0, ".")
from wallgraph.detector import WallDetector
from wallgraph.builder import WallGraphBuilder

IMAGE = "sample/raw/image-3.jpg"

detector = WallDetector()
walls = detector.detect(IMAGE)

print(f"=== Detected walls: {len(walls)} ===")
straight = [w for w in walls if not w.is_arc]
arcs = [w for w in walls if w.is_arc]
print(f"Straight walls: {len(straight)}")
for i, w in enumerate(straight):
    orient = "H" if w.is_horizontal() else ("V" if w.is_vertical() else "D")
    print(f"  W{i:2d}: ({w.start.x:.0f},{w.start.y:.0f})→({w.end.x:.0f},{w.end.y:.0f}) {orient} {w.length():.0f}px")

print(f"Arc walls: {len(arcs)}")
for i, w in enumerate(arcs):
    print(f"  A{i:2d}: center=({w.center.x:.0f},{w.center.y:.0f}) r={w.radius:.0f} span={abs(w.end_angle-w.start_angle)*180/3.14159:.0f}°")

# Visualize
vis = detector.visualize(IMAGE, walls, "debug_img3_new.png")

# Also show what the wall_mask looks like
img = cv2.imread(IMAGE)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# Small component removal
cleaned = np.zeros_like(binary)
num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
for i in range(1, num_labels):
    if stats[i, cv2.CC_STAT_AREA] >= 50:
        mask = (labels == i).astype(np.uint8) * 255
        cleaned = cv2.bitwise_or(cleaned, mask)

# Heavy open (wall mask)
k_heavy = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
wall_mask = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, k_heavy)

# Light open (original)
k_light = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
light_mask = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, k_light)

# Text mask: what light_open sees but heavy_open doesn't
text_mask = cv2.subtract(light_mask, wall_mask)

cv2.imwrite("debug_img3_wall_mask.png", wall_mask)
cv2.imwrite("debug_img3_text_mask.png", text_mask)
cv2.imwrite("debug_img3_cleaned.png", cleaned)

print("\nSaved debug files: debug_img3_new.png, debug_img3_wall_mask.png, debug_img3_text_mask.png, debug_img3_cleaned.png")

# Build WallGraph
builder = WallGraphBuilder()
graph = builder.build(walls)
print(f"\n=== WallGraph: {len(graph.walls)} walls ===")

from wallgraph.validator import WallGraphValidator
validator = WallGraphValidator()
report = validator.validate(graph)
print(report)
