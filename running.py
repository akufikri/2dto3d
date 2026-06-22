#!/usr/bin/env python3
"""
FloorPlan 3D — wall extraction engine.

Outputs THREE.js-ready JSON (shapes + holes for ExtrudeGeometry).
Frontend consumes the JSON separately.

Usage:
  python3 running.py image.png --api               # ML mask → JSON to stdout
  python3 running.py image.png --api -o out.json   # save to file
  python3 running.py image.png --mask mask.png     # use saved ML mask
  python3 running.py image.png                     # Hough on raw image (fallback)
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from wallgraph.detector import WallDetector
from wallgraph.builder import WallGraphBuilder
from wallgraph.validator import WallGraphValidator

SAMPLE_DIR = ROOT / "sample" / "raw"
IMG_EXTS   = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}

DEFAULT_API = "http://localhost:8000"


# ── ML mask fetch ─────────────────────────────────────────────────────────────

def fetch_mask(image_path: Path, api_url: str, preprocess: str | None = None) -> Path:
    """Call the ML API and save wall mask to a temp file. Returns path."""
    try:
        import requests
    except ImportError:
        print("[api]  'requests' not installed. Run: pip install requests", file=sys.stderr)
        sys.exit(1)

    mask_path = ROOT / "output" / f"wall_mask_{image_path.stem}.png"
    mask_path.parent.mkdir(exist_ok=True)

    params: dict = {"max_dim": 1024}
    if preprocess:
        params["preprocess"] = preprocess

    print(f"[api]   fetching mask from {api_url} ...", file=sys.stderr)
    try:
        with open(image_path, "rb") as f:
            resp = requests.post(
                f"{api_url}/predict/wall-mask",
                files={"file": (image_path.name, f)},
                params=params,
                timeout=120,
            )
    except Exception as e:
        print(f"[api]   connection error: {e}", file=sys.stderr)
        sys.exit(1)

    if resp.status_code != 200:
        print(f"[api]   error {resp.status_code}: {resp.text}", file=sys.stderr)
        sys.exit(1)

    mask_path.write_bytes(resp.content)
    server_time = resp.headers.get("X-Inference-Time", "?")
    print(f"[api]   mask saved → {mask_path}  (server: {server_time})", file=sys.stderr)
    return mask_path


# ── Detection pipeline ────────────────────────────────────────────────────────

def run_pipeline(
    image_path: Path,
    mask_path: Path | None = None,
    snap: float = 8.0,
    min_length: float = 35.0,
) -> dict:
    detector = WallDetector(params={"min_wall_length": min_length})

    if mask_path is not None:
        print(f"[detect] mask  → {mask_path.name}", file=sys.stderr)
        walls, doors = detector.detect_from_mask(str(mask_path))
    else:
        print(f"[detect] image → {image_path.name}", file=sys.stderr)
        walls, doors = detector.detect_with_doors(str(image_path))

    print(f"[detect] raw walls={len(walls)}  doors={len(doors)}", file=sys.stderr)

    builder = WallGraphBuilder(snap_distance=snap, min_wall_length=min_length)
    graph   = builder.build(walls)
    graph.doors = doors

    validator = WallGraphValidator(snap_distance=snap)
    report    = validator.validate(graph)
    print(report, file=sys.stderr)

    return json.loads(graph.model_dump_json())


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="FloorPlan 3D — wall extraction engine (outputs THREE.js JSON)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  python3 running.py image.png --api              # best quality (ML mask) → stdout
  python3 running.py image.png --api -o out.json  # save to file
  python3 running.py image.png --mask m.png       # use saved mask
  python3 running.py image.png                    # raw image Hough (fallback)
""",
    )
    parser.add_argument(
        "image", type=Path,
        help="Path to floorplan image"
    )
    parser.add_argument(
        "--mask", type=Path, metavar="MASK",
        help="Path to binary wall mask PNG (white=wall). Skips raw image detection."
    )
    parser.add_argument(
        "--api", nargs="?", const=DEFAULT_API, metavar="URL",
        help=f"Fetch ML wall mask from API (default: {DEFAULT_API})"
    )
    parser.add_argument(
        "--preprocess",
        choices=["furniture", "annotation", "both",
                 "yolo_annotation", "yolo_filter", "yolo_full"],
        default=None,
        help=(
            "Preprocessing mode when fetching mask from API. "
            "yolo_full = both YOLO steps (best quality)."
        )
    )
    parser.add_argument(
        "-o", "--output", type=Path,
        help="Save JSON to file (default: print to stdout)"
    )
    parser.add_argument(
        "--visualize", type=Path, metavar="STEM",
        help=(
            "Save debug images: STEM_skeleton.png (wall outlines) and "
            "STEM_connector.png (wall outlines + vertex dots). "
            "Example: --visualize output/debug_image-4"
        )
    )
    parser.add_argument(
        "--snap", type=float, default=None,
        help="Endpoint snap distance px (default: 12 with --mask/--api, 8 otherwise)"
    )
    parser.add_argument(
        "--min-length", type=float, default=None,
        help="Minimum wall length px (default: 25 with --mask/--api, 35 otherwise)"
    )
    parser.add_argument(
        "--contour", action="store_true",
        help=(
            "Use contour-based extraction instead of Hough line detection. "
            "Produces exact wall shapes from the mask — no gaps, no missed corners. "
            "Requires --mask or --api. Outputs 'shape' JSON format for ExtrudeGeometry."
        )
    )
    parser.add_argument(
        "--epsilon", type=float, default=4.0,
        help="Polygon simplification (px) for --contour mode. Larger = smoother walls. Default: 4.0"
    )
    parser.add_argument(
        "--close", type=int, default=5, metavar="PX",
        help="Morphological closing kernel radius (px) to fill corner/junction gaps. Default: 5"
    )
    parser.add_argument(
        "--smooth", type=int, default=3, metavar="PX",
        help="Gaussian blur radius for edge smoothing before contour extraction. Default: 3"
    )
    parser.add_argument(
        "--dilate", type=int, default=0,
        help="Extra dilation after close/smooth (px). Default: 0"
    )
    parser.add_argument(
        "--no-snap", action="store_true",
        help="Disable axis snapping (H/V edge straightening) in --contour mode"
    )
    parser.add_argument(
        "--snap-tol", type=float, default=25.0, metavar="DEG",
        help="Angle tolerance for H/V axis snapping (degrees, default: 25)"
    )
    parser.add_argument(
        "--min-area", type=int, default=300, metavar="PX2",
        help="Min contour area px² to keep (filters furniture noise, default: 300)"
    )
    parser.add_argument(
        "--notch-depth", type=float, default=0.0, metavar="PX",
        help=(
            "Remove polygon vertices where deviation from neighbor line < PX. "
            "Cleans window notches/bumps on straight walls. Typical: 15-25. Default: 0 (off)"
        )
    )
    parser.add_argument(
        "--min-extent", type=int, default=0, metavar="PX",
        help=(
            "Min bounding box longer side (px) to keep contour. "
            "Filters small blobby fragments without removing thin walls. "
            "Typical: 40-60. Default: 0 (disabled)"
        )
    )
    parser.add_argument(
        "--wall-thickness", type=int, default=0, metavar="PX",
        help=(
            "Normalize all walls to uniform half-thickness (px) via skeleton → re-dilation. "
            "Fixes uneven wall widths from ML mask. Typical: 8 (→16px walls). Default: 0 (off)"
        )
    )

    # ── Opening detection ──────────────────────────────────────────────────
    parser.add_argument(
        "--detect-openings", action="store_true",
        help=(
            "Detect doors and windows geometrically from wall mask gaps. "
            "Finds sandwiched black regions in the mask before closing. "
            "Only works with --contour mode."
        )
    )
    parser.add_argument(
        "--door-mask", type=Path, metavar="MASK",
        help=(
            "Path to binary door mask PNG (white=door). "
            "From extended CubiCasa API /predict/door-mask. "
            "Overrides geometric door detection."
        )
    )
    parser.add_argument(
        "--window-mask", type=Path, metavar="MASK",
        help=(
            "Path to binary window mask PNG (white=window). "
            "From extended CubiCasa API /predict/window-mask. "
            "Overrides geometric window detection."
        )
    )

    args = parser.parse_args()

    if not args.image.exists():
        print(f"Error: file not found: {args.image}", file=sys.stderr)
        sys.exit(1)

    # Resolve mask
    mask_path: Path | None = args.mask
    if args.api and mask_path is None:
        mask_path = fetch_mask(args.image, args.api, preprocess=args.preprocess)

    # Auto-select better defaults when working from a clean ML mask
    using_mask = mask_path is not None
    snap       = args.snap       if args.snap       is not None else (12.0 if using_mask else 8.0)
    min_length = args.min_length if args.min_length is not None else (25.0 if using_mask else 35.0)

    # Run pipeline
    if args.contour:
        if mask_path is None:
            print("[error] --contour requires --mask or --api (needs a binary wall mask)", file=sys.stderr)
            sys.exit(1)
        from wallgraph.mask_to_shape import extract_wall_shapes
        print(
            f"[contour] extracting polygons from {mask_path.name} "
            f"(epsilon={args.epsilon}, close={args.close}, smooth={args.smooth}, dilate={args.dilate})",
            file=sys.stderr,
        )
        graph_data = extract_wall_shapes(
            str(mask_path),
            epsilon=args.epsilon,
            min_area=args.min_area,
            close_px=args.close,
            smooth_blur=args.smooth,
            dilate_px=args.dilate,
            snap_axes=not args.no_snap,
            snap_tol_deg=args.snap_tol,
            uniform_thickness=args.wall_thickness,
            min_extent=args.min_extent,
            notch_depth=args.notch_depth,
            detect_openings=args.detect_openings,
            door_mask_path=str(args.door_mask) if args.door_mask else None,
            window_mask_path=str(args.window_mask) if args.window_mask else None,
        )
        n_shapes  = len(graph_data["shapes"])
        n_holes   = sum(len(s["holes"]) for s in graph_data["shapes"])
        n_doors   = len(graph_data.get("doors", []))
        n_windows = len(graph_data.get("windows", []))
        print(f"[contour] {n_shapes} wall shape(s), {n_holes} room hole(s)", file=sys.stderr)
        if n_doors or n_windows:
            print(f"[openings] {n_doors} door(s), {n_windows} window(s)", file=sys.stderr)
    else:
        graph_data = run_pipeline(
            args.image,
            mask_path=mask_path,
            snap=snap,
            min_length=min_length,
        )

    # Debug visualization
    if args.visualize and args.contour:
        from wallgraph.visualizer import render_both
        skel, conn = render_both(str(args.visualize), graph_data)
        print(f"[vis]    skeleton  → {skel}", file=sys.stderr)
        print(f"[vis]    connector → {conn}", file=sys.stderr)

    # Output
    output_json = json.dumps(graph_data, indent=2)
    if args.output:
        args.output.write_text(output_json)
        print(f"[output] saved → {args.output}", file=sys.stderr)
    else:
        print(output_json)


if __name__ == "__main__":
    main()
