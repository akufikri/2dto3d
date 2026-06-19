#!/usr/bin/env python3
"""Reproduce component wall mask and compare old vs new classification."""

import cv2, numpy as np, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wallgraph import WallDetector

IMAGE = "sample/raw/17817183169a88.png"
OUT = "sample/episodes/episode-8"

detector = WallDetector()
img = cv2.imread(IMAGE)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Pipeline up to cleaned mask
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

# NEW classification
wall_mask_new = detector._build_wall_mask(cleaned)
cv2.imwrite(f"{OUT}/debug_component_wall_mask_new.png", wall_mask_new)

# Analyze components
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(cleaned, 8)
min_dim_threshold = detector.params["wall_min_dim"]
solidity_threshold = detector.params["wall_solidity"]
max_aspect = detector.params["wall_max_aspect"]
large_area_threshold = detector.params["min_component_area"] * detector.params["wall_large_area_mult"]

# Old classification params for comparison
old_min_dim = 15
old_area_threshold = detector.params["min_component_area"] * 4

print(f"Image: {img.shape}")
print(f"Total components: {num_labels - 1}")
print(f"NEW thresholds: min_dim={min_dim_threshold}, solidity={solidity_threshold}, "
      f"max_aspect={max_aspect}, large_area={large_area_threshold}")
print(f"OLD thresholds: min_dim={old_min_dim}, area={old_area_threshold}")
print()

# Classification comparison visualization
debug_vis = cv2.cvtColor(cleaned, cv2.COLOR_GRAY2BGR)
changed = []  # components that changed classification

for i in range(1, num_labels):
    comp_area = stats[i, cv2.CC_STAT_AREA]
    w = stats[i, cv2.CC_STAT_WIDTH]
    h_c = stats[i, cv2.CC_STAT_HEIGHT]
    min_dim = min(w, h_c)
    max_dim = max(w, h_c)
    bbox_area = w * h_c
    solidity = comp_area / bbox_area if bbox_area > 0 else 0.0
    aspect = max_dim / max(min_dim, 1)

    # Old classification
    old_is_wall = min_dim >= old_min_dim or comp_area >= old_area_threshold

    # New classification
    new_is_wall = (
        (comp_area >= large_area_threshold)
        or (min_dim >= min_dim_threshold and solidity >= solidity_threshold and aspect <= max_aspect)
    )

    label_old = "WALL" if old_is_wall else "TEXT"
    label_new = "WALL" if new_is_wall else "TEXT"
    changed_flag = " *** CHANGED" if old_is_wall != new_is_wall else ""

    print(f"  comp{i}: area={comp_area}, bbox=({stats[i,cv2.CC_STAT_LEFT]},{stats[i,cv2.CC_STAT_TOP]},{w},{h_c}), "
          f"min_dim={min_dim}, max_dim={max_dim}, solidity={solidity:.3f}, aspect={aspect:.1f} "
          f"→ old={label_old}, new={label_new}{changed_flag}")

    mask_i = (labels == i)
    if new_is_wall:
        debug_vis[mask_i] = [0, 255, 0]  # green = wall
    else:
        debug_vis[mask_i] = [0, 0, 255]  # red = noise

    if old_is_wall != new_is_wall:
        changed.append(i)

cv2.imwrite(f"{OUT}/debug_component_classification_new.png", debug_vis)

print(f"\nComponents reclassified from WALL→TEXT: {len(changed)}")
for i in changed:
    comp_area = stats[i, cv2.CC_STAT_AREA]
    w = stats[i, cv2.CC_STAT_WIDTH]
    h_c = stats[i, cv2.CC_STAT_HEIGHT]
    min_dim = min(w, h_c)
    max_dim = max(w, h_c)
    bbox_area = w * h_c
    solidity = comp_area / bbox_area if bbox_area > 0 else 0.0
    aspect = max_dim / max(min_dim, 1)
    print(f"  comp{i}: area={comp_area}, min_dim={min_dim}, max_dim={max_dim}, "
          f"solidity={solidity:.3f}, aspect={aspect:.1f}")

print(f"\nSaved: {OUT}/debug_component_wall_mask_new.png")
print(f"Saved: {OUT}/debug_component_classification_new.png")
print("Green = WALL, Red = TEXT/noise")
