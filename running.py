#!/usr/bin/env python3
"""
FloorPlan 3D — single entry point.

Recommended pipeline (best quality):
  python3 running.py image.png --api http://167.172.88.109:8000

  This fetches a clean ML wall mask from the API, skeletonizes it to
  centerlines, then builds the wall graph — far fewer spurious walls than
  running Hough directly on the noisy raw image.

Other modes:
  python3 running.py                               # pick sample interactively
  python3 running.py image.png                     # Hough on raw image (fallback)
  python3 running.py image.png --mask mask.png     # use saved ML mask
  python3 running.py image.png --api URL           # fetch mask from API then detect
  python3 running.py image.png --cli               # print JSON to stdout
  python3 running.py image.png -o out.json         # save JSON, no browser
  python3 running.py image.png --port 9000         # custom viewer port
"""

import argparse
import http.server
import json
import os
import sys
import webbrowser
from pathlib import Path

# ── Project root on sys.path ──────────────────────────────────────────────────
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from wallgraph.detector import WallDetector
from wallgraph.builder import WallGraphBuilder
from wallgraph.validator import WallGraphValidator

SAMPLE_DIR = ROOT / "sample" / "raw"
VIEWER_DIR = ROOT / "viewer"
IMG_EXTS   = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}

DEFAULT_API = "http://167.172.88.109:8000"


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


# ── Interactive sample picker ─────────────────────────────────────────────────

def pick_sample() -> Path | None:
    samples = sorted(
        [p for p in SAMPLE_DIR.iterdir() if p.suffix.lower() in IMG_EXTS],
        key=lambda p: p.name,
    )
    if not samples:
        print(f"No images found in {SAMPLE_DIR}", file=sys.stderr)
        return None

    print("\nAvailable samples:")
    for i, p in enumerate(samples, 1):
        size = p.stat().st_size
        sz   = f"{size/1024:.0f} KB" if size < 1_000_000 else f"{size/1e6:.1f} MB"
        print(f"  [{i:2}] {p.name}  ({sz})")
    print("  [ 0] quit\n")

    while True:
        try:
            raw = input("Pick number: ").strip()
        except (KeyboardInterrupt, EOFError):
            return None
        if raw == "0":
            return None
        try:
            idx = int(raw) - 1
            if 0 <= idx < len(samples):
                return samples[idx]
        except ValueError:
            pass
        print("Invalid choice.")


# ── HTTP server for viewer ────────────────────────────────────────────────────

class _QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *_):
        pass


def serve_viewer(port: int, graph_data: dict) -> None:
    graph_path = VIEWER_DIR / "graph.json"
    graph_path.write_text(json.dumps(graph_data, indent=2))

    os.chdir(VIEWER_DIR)
    server = http.server.HTTPServer(("", port), _QuietHandler)
    url    = f"http://localhost:{port}?json=graph.json"

    print(f"\n[viewer] serving at {url}")
    print(f"[viewer] Ctrl+C to stop\n")
    webbrowser.open(url)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        graph_path.unlink(missing_ok=True)
        print("\n[viewer] stopped.")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="FloorPlan 3D — detect walls and view in 3D",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  python3 running.py image.png --api          # best quality (ML mask)
  python3 running.py image.png --mask m.png   # use saved mask
  python3 running.py image.png                # raw image Hough (fallback)
  python3 running.py image.png --cli          # JSON to stdout
""",
    )
    parser.add_argument(
        "image", nargs="?", type=Path,
        help="Path to floorplan image (omit for interactive sample picker)"
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
        "--preprocess", choices=["furniture", "annotation", "both"], default=None,
        help="Preprocessing mode when fetching mask from API"
    )
    parser.add_argument(
        "--cli", action="store_true",
        help="Print WallGraph JSON to stdout instead of opening viewer"
    )
    parser.add_argument(
        "-o", "--output", type=Path,
        help="Save WallGraph JSON to file (skips browser)"
    )
    parser.add_argument(
        "--port", type=int, default=8765,
        help="HTTP port for 3D viewer (default: 8765)"
    )
    parser.add_argument(
        "--snap", type=float, default=None,
        help="Endpoint snap distance px (default: 12 with --mask/--api, 8 otherwise)"
    )
    parser.add_argument(
        "--min-length", type=float, default=None,
        help="Minimum wall length px (default: 25 with --mask/--api, 35 otherwise)"
    )
    args = parser.parse_args()

    # Resolve image path
    image_path: Path | None = args.image
    if image_path is None:
        image_path = pick_sample()
        if image_path is None:
            sys.exit(0)

    if not image_path.exists():
        print(f"Error: file not found: {image_path}", file=sys.stderr)
        sys.exit(1)

    # Resolve mask
    mask_path: Path | None = args.mask
    if args.api and mask_path is None:
        mask_path = fetch_mask(image_path, args.api, preprocess=args.preprocess)

    # Auto-select better defaults when working from a clean ML mask
    using_mask = mask_path is not None
    snap       = args.snap       if args.snap       is not None else (12.0 if using_mask else 8.0)
    min_length = args.min_length if args.min_length is not None else (25.0 if using_mask else 35.0)

    # Run pipeline
    graph_data = run_pipeline(
        image_path,
        mask_path=mask_path,
        snap=snap,
        min_length=min_length,
    )

    # Output
    if args.cli:
        print(json.dumps(graph_data, indent=2))
        return

    if args.output:
        args.output.write_text(json.dumps(graph_data, indent=2))
        print(f"[output] saved → {args.output}", file=sys.stderr)
        return

    serve_viewer(args.port, graph_data)


if __name__ == "__main__":
    main()
