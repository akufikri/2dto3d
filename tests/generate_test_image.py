#!/usr/bin/env python3
"""Generate synthetic floorplan for testing wall detection."""

import cv2
import numpy as np


def generate_floorplan(width=800, height=600, output="sample/test_floorplan.png"):
    img = np.ones((height, width, 3), dtype=np.uint8) * 255

    def draw_wall(x1, y1, x2, y2, thickness=8):
        """Draw thick wall line (double line style)."""
        cv2.rectangle(img, (x1 - thickness // 2, y1 - thickness // 2),
                      (x2 + thickness // 2, y2 + thickness // 2), (0, 0, 0), -1)
        # Draw as thick line instead
        cv2.line(img, (x1, y1), (x2, y2), (0, 0, 0), thickness)
        cv2.line(img, (x1, y1 - thickness // 2), (x2, y2 - thickness // 2), (0, 0, 0), 2)
        cv2.line(img, (x1, y1 + thickness // 2), (x2, y2 + thickness // 2), (0, 0, 0), 2)

    # Outer walls (rectangle)
    cv2.rectangle(img, (50, 50), (750, 550), (0, 0, 0), 10)

    # Inner dividing wall (vertical)
    cv2.rectangle(img, (400, 50), (410, 350), (0, 0, 0), 8)

    # Inner dividing wall (horizontal)
    cv2.rectangle(img, (50, 300), (400, 310), (0, 0, 0), 8)

    # Door opening (gap in bottom wall)
    cv2.rectangle(img, (300, 545), (380, 555), (255, 255, 255), -1)

    # Window (3 lines style)
    win_y = 50 + 30
    for offset in [0, 4, 8]:
        cv2.line(img, (450, win_y + offset), (550, win_y + offset), (100, 100, 100), 2)

    # Room label simulation
    cv2.putText(img, "RM1", (80, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 200, 200), 2)
    cv2.putText(img, "RM2", (450, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 200, 200), 2)

    cv2.imwrite(output, img)
    print(f"Test floorplan saved: {output}")
    return img


if __name__ == "__main__":
    generate_floorplan()
