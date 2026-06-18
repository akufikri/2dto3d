import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print("imports starting...")
import cv2, numpy as np
print("cv2 ok")
from wallgraph.detector import WallDetector
print("detector ok")
from wallgraph.models import Wall, Point
print("all imports done")

print("reading image...")
img = cv2.imread("sample/image-6.jpeg")
print(f"read done: {img.shape}")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
print(f"threshold done. binary nnz={cv2.countNonZero(binary)}")

detector = WallDetector()
print("detector inited")

cleaned = detector._remove_small_components(binary)
print(f"clean done: {cv2.countNonZero(cleaned)}")

skel = detector._skeletonize(cleaned)
print(f"skeleton done: {cv2.countNonZero(skel)}")

lines = cv2.HoughLinesP(skel, 1, np.pi/180, threshold=50, minLineLength=50, maxLineGap=15)
print(f"hough done: {len(lines) if lines is not None else 0} lines")
