"""
Shared pytest fixtures for wallgraph tests.

Provides:
- ``simple_graph`` — a minimal 4-wall closed rectangle graph (no image needed)
- ``synthetic_image`` — an OpenCV image array of a simple floorplan
- ``synthetic_image_path`` — path to a saved synthetic floorplan PNG
"""

from __future__ import annotations

import os
import tempfile

import cv2
import numpy as np
import pytest

from wallgraph.models import Point, Wall, WallGraph


# ---------------------------------------------------------------------------
# Model fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def simple_rectangle_walls() -> list[Wall]:
    """Four walls forming a 400×300 closed rectangle."""
    return [
        Wall(id="W0000", start=Point(x=100, y=100), end=Point(x=500, y=100)),  # top
        Wall(id="W0001", start=Point(x=500, y=100), end=Point(x=500, y=400)),  # right
        Wall(id="W0002", start=Point(x=500, y=400), end=Point(x=100, y=400)),  # bottom
        Wall(id="W0003", start=Point(x=100, y=400), end=Point(x=100, y=100)),  # left
    ]


@pytest.fixture
def simple_graph(simple_rectangle_walls) -> WallGraph:
    """WallGraph containing the simple rectangle."""
    return WallGraph(walls=simple_rectangle_walls)


@pytest.fixture
def t_junction_walls() -> list[Wall]:
    """Three walls forming a T-junction at (300, 250)."""
    return [
        # Horizontal base: (100,250) → (500,250)
        Wall(id="W0000", start=Point(x=100, y=250), end=Point(x=500, y=250)),
        # Vertical stem: (300,100) → (300,250)  [end touches base]
        Wall(id="W0001", start=Point(x=300, y=100), end=Point(x=300, y=250)),
    ]


# ---------------------------------------------------------------------------
# Image fixtures
# ---------------------------------------------------------------------------


def _draw_synthetic_floorplan(width: int = 600, height: int = 400) -> np.ndarray:
    """Return an OpenCV BGR image with a simple L-shaped floorplan."""
    img = np.ones((height, width, 3), dtype=np.uint8) * 255
    wall_color = (0, 0, 0)
    thickness = 8

    # Outer rectangle
    cv2.rectangle(img, (50, 50), (550, 350), wall_color, thickness)

    # Interior dividing wall (vertical)
    cv2.line(img, (300, 50), (300, 250), wall_color, thickness)

    # Interior dividing wall (horizontal)
    cv2.line(img, (50, 250), (300, 250), wall_color, thickness)

    return img


@pytest.fixture
def synthetic_image() -> np.ndarray:
    """In-memory OpenCV BGR image of a synthetic floorplan."""
    return _draw_synthetic_floorplan()


@pytest.fixture
def synthetic_image_path(tmp_path) -> str:
    """Path to a saved synthetic floorplan PNG (auto-cleaned by pytest)."""
    img = _draw_synthetic_floorplan()
    path = str(tmp_path / "test_floorplan.png")
    cv2.imwrite(path, img)
    return path
