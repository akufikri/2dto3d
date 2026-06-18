#!/usr/bin/env python3
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import time, cv2, numpy as np
from wallgraph.detector import WallDetector
from wallgraph.builder import WallGraphBuilder
from wallgraph.validator import WallGraphValidator
from wallgraph.utils import merge_collinear_segments
from wallgraph.models import Wall, Point

detector = WallDetector(params={"min_wall_length": 50, "min_component_area": 50})

img = cv2.imread("sample/image-6.jpeg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
t0 = time.time()
_, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
print(f"threshold: {time.time()-t0:.3f}s")

t0 = time.time()
cleaned = detector._remove_small_components(binary)
print(f"remove_small: {time.time()-t0:.3f}s")

t0 = time.time()
skel = detector._skeletonize(cleaned)
print(f"skeletonize: {time.time()-t0:.3f}s")

t0 = time.time()
lines = cv2.HoughLinesP(skel, 1, np.pi/180, threshold=50, minLineLength=50, maxLineGap=15)
print(f"Hough: {time.time()-t0:.3f}s -> {len(lines) if lines is not None else 0}")

raw = []
if lines is not None:
    for line in lines:
        x1,y1,x2,y2 = line[0]
        raw.append(((float(x1),float(y1)),(float(x2),float(y2))))

t0 = time.time()
snapped = detector._snap_to_axis(raw)
print(f"snap: {time.time()-t0:.3f}s")

t0 = time.time()
merged = merge_collinear_segments(snapped, angle_tol=np.radians(3), dist_tol=5.0)
print(f"merge_collinear: {time.time()-t0:.3f}s -> {len(merged)}")

walls = []
for i, ((x1,y1),(x2,y2)) in enumerate(merged):
    walls.append(Wall(id=f"W{i:04d}", start=Point(x=x1,y=y1), end=Point(x=x2,y=y2)))
t0 = time.time()
walls = detector._merge_parallel_pairs(walls)
print(f"merge_parallel: {time.time()-t0:.3f}s -> {len(walls)}")
walls = [w for w in walls if w.length() >= 50]
print(f"final walls: {len(walls)}")

t0 = time.time()
builder = WallGraphBuilder(snap_distance=5.0, min_wall_length=50)
graph = builder.build(walls)
print(f"builder: {time.time()-t0:.3f}s -> {len(graph.walls)} walls")

v = WallGraphValidator()
r = v.validate(graph)
print(r)
