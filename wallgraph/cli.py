#!/usr/bin/env python3
"""WallGraph CLI — Detect walls from floorplan image and build wall graph."""

import argparse
import json
import sys
import os

from .detector import WallDetector
from .builder import WallGraphBuilder
from .validator import WallGraphValidator


def main():
    parser = argparse.ArgumentParser(description="Wall-First CAD Detection Pipeline")
    parser.add_argument("image", help="Path to floorplan image")
    parser.add_argument("--output", "-o", help="Output JSON path (default: stdout)")
    parser.add_argument("--visualize", "-v", help="Output debug visualization path")
    parser.add_argument(
        "--method",
        choices=["binary", "canny", "contour"],
        default="binary",
        help="Detection method (default: binary)",
    )
    parser.add_argument(
        "--snap", type=float, default=5.0, help="Endpoint snap distance (default: 5.0)"
    )
    parser.add_argument(
        "--min-length", type=float, default=40, help="Min wall length in pixels"
    )

    args = parser.parse_args()

    if not os.path.exists(args.image):
        print(f"Error: Image not found: {args.image}", file=sys.stderr)
        sys.exit(1)

    detector = WallDetector(params={
        "min_wall_length": args.min_length,
        "snap_distance": args.snap,
    })

    print(f"Detecting walls from: {args.image}", file=sys.stderr)

    if args.method == "contour":
        walls = detector.detect_contour(args.image)
    elif args.method == "canny":
        walls = detector.detect_canny(args.image)
    else:
        walls = detector.detect(args.image)

    print(f"Raw walls detected: {len(walls)}", file=sys.stderr)

    builder = WallGraphBuilder(snap_distance=args.snap, min_wall_length=args.min_length)
    graph = builder.build(walls)

    validator = WallGraphValidator(snap_distance=args.snap)
    report = validator.validate(graph)
    print(report, file=sys.stderr)

    if args.visualize:
        detector.visualize(args.image, graph.walls, args.visualize)
        print(f"Visualization saved: {args.visualize}", file=sys.stderr)

    output = graph.model_dump_json(indent=2)
    if args.output:
        with open(args.output, "w") as f:
            f.write(output)
        print(f"Wall graph saved: {args.output}", file=sys.stderr)
    else:
        print(output)


if __name__ == "__main__":
    main()
