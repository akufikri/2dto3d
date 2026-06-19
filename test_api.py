#!/usr/bin/env python3
"""
FloorPlan Wall Mask API - CLI Tester
Usage:
  python test_api.py                              # interactive menu
  python test_api.py -f path/to/img.jpg          # langsung test file
  python test_api.py -f img.jpg --preprocess furniture  # hapus furniture dulu
  python test_api.py --url http://...             # custom API URL
"""

import argparse
import os
import sys
import time
from pathlib import Path

import requests
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import print as rprint

console = Console()

API_URL = "http://167.172.88.109:8000"
SAMPLE_DIR = Path(__file__).parent / "sample" / "raw"
OUTPUT_DIR = Path(__file__).parent / "output"

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}


def check_health(base_url: str) -> bool:
    try:
        r = requests.get(f"{base_url}/health", timeout=5)
        data = r.json()
        console.print(
            f"[green]✓ API online[/green] — model: [bold]{data['model']}[/bold], "
            f"epoch: [bold]{data['epoch']}[/bold]"
        )
        return True
    except Exception as e:
        console.print(f"[red]✗ API offline:[/red] {e}")
        return False


def get_sample_images() -> list[Path]:
    if not SAMPLE_DIR.exists():
        return []
    return sorted(
        [p for p in SAMPLE_DIR.iterdir() if p.suffix.lower() in IMG_EXTS],
        key=lambda p: p.name,
    )


def predict(
    image_path: Path,
    base_url: str,
    preprocess: str | None = None,
    max_dim: int = 1024,
) -> Path | None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    suffix = f"_{preprocess}" if preprocess else ""
    out_path = OUTPUT_DIR / f"wall_mask_{image_path.stem}{suffix}.png"

    pre_label = f" [cyan]+{preprocess}[/cyan]" if preprocess else ""
    console.print(f"\n[dim]Uploading:[/dim] {image_path.name}{pre_label}")
    t0 = time.time()

    params: dict = {"max_dim": max_dim}
    if preprocess:
        params["preprocess"] = preprocess

    try:
        with open(image_path, "rb") as f:
            resp = requests.post(
                f"{base_url}/predict/wall-mask",
                files={"file": (image_path.name, f)},
                params=params,
                timeout=60,
            )
    except requests.exceptions.ConnectionError:
        console.print("[red]✗ Cannot connect to API[/red]")
        return None

    elapsed = time.time() - t0

    if resp.status_code != 200:
        console.print(f"[red]✗ API error {resp.status_code}:[/red] {resp.text}")
        return None

    server_time = resp.headers.get("X-Inference-Time", "?")
    pre_applied = resp.headers.get("X-Preprocess", "none")

    out_path.write_bytes(resp.content)
    size_kb = len(resp.content) / 1024

    console.print(
        f"[green]✓ Done[/green] — "
        f"server: [bold]{server_time}[/bold], "
        f"total: [bold]{elapsed:.2f}s[/bold], "
        f"mask: [bold]{size_kb:.1f}KB[/bold], "
        f"preprocess: [bold]{pre_applied}[/bold]"
    )
    console.print(f"[dim]Saved →[/dim] {out_path}")
    return out_path


def select_sample_menu(samples: list[Path]) -> Path | None:
    table = Table(show_header=False, box=None, padding=(0, 1))
    table.add_column("No", style="bold cyan", width=4)
    table.add_column("File")
    table.add_column("Size", style="dim", justify="right")

    for i, p in enumerate(samples, 1):
        size = p.stat().st_size
        size_str = f"{size/1024:.0f}KB" if size < 1_000_000 else f"{size/1_000_000:.1f}MB"
        table.add_row(str(i), p.name, size_str)

    console.print(table)
    console.print("[dim]0 = kembali[/dim]")

    while True:
        try:
            choice = input("\nPilih nomor: ").strip()
            if choice == "0":
                return None
            idx = int(choice) - 1
            if 0 <= idx < len(samples):
                return samples[idx]
            console.print("[yellow]Nomor tidak valid[/yellow]")
        except (ValueError, KeyboardInterrupt):
            return None


def interactive_mode(base_url: str):
    console.print(
        Panel(
            "[bold]FloorPlan Wall Mask API Tester[/bold]\n"
            f"[dim]{base_url}[/dim]",
            expand=False,
        )
    )

    if not check_health(base_url):
        sys.exit(1)

    while True:
        console.print("\n[bold]Menu:[/bold]")
        console.print("  [cyan]1[/cyan] Pilih dari sample")
        console.print("  [cyan]2[/cyan] Input path sendiri")
        console.print("  [cyan]q[/cyan] Keluar")

        try:
            choice = input("\n> ").strip().lower()
        except KeyboardInterrupt:
            console.print("\n[dim]Bye.[/dim]")
            break

        if choice == "q":
            console.print("[dim]Bye.[/dim]")
            break

        elif choice == "1":
            samples = get_sample_images()
            if not samples:
                console.print(f"[yellow]Tidak ada gambar di[/yellow] {SAMPLE_DIR}")
                continue
            console.print(f"\n[bold]Sample images[/bold] ({len(samples)} file):\n")
            img = select_sample_menu(samples)
            if img:
                pre = _ask_preprocess()
                out = predict(img, base_url, preprocess=pre)
                if out:
                    _offer_open(out)

        elif choice == "2":
            try:
                raw = input("Path gambar: ").strip().strip("'\"")
            except KeyboardInterrupt:
                continue
            img = Path(raw).expanduser()
            if not img.exists():
                console.print(f"[red]File tidak ditemukan:[/red] {img}")
                continue
            if img.suffix.lower() not in IMG_EXTS:
                console.print(f"[yellow]Extension tidak dikenal:[/yellow] {img.suffix}")
            pre = _ask_preprocess()
            out = predict(img, base_url, preprocess=pre)
            if out:
                _offer_open(out)

        else:
            console.print("[yellow]Pilihan tidak valid[/yellow]")


def _ask_preprocess() -> str | None:
    console.print("\n[dim]Preprocess? [[N]/furniture/annotation/both][/dim]", end=" ")
    try:
        ans = input().strip().lower()
    except KeyboardInterrupt:
        return None
    return ans if ans in ("furniture", "annotation", "both") else None


def _offer_open(path: Path):
    try:
        ans = input("Buka gambar? [y/N] ").strip().lower()
    except (KeyboardInterrupt, EOFError):
        return
    if ans == "y":
        os.system(f'open "{path}"')


def main():
    parser = argparse.ArgumentParser(description="FloorPlan Wall Mask API Tester")
    parser.add_argument("-f", "--file", type=Path, help="Path gambar untuk ditest")
    parser.add_argument("--url", default=API_URL, help=f"API base URL (default: {API_URL})")
    parser.add_argument(
        "--preprocess", choices=["furniture", "annotation", "both"], default=None,
        help=(
            "furniture = hapus furniture hitam (gray-wall plans) | "
            "annotation = hapus text/hatching/anotasi kecil | "
            "both = keduanya"
        )
    )
    parser.add_argument(
        "--max-dim", type=int, default=1024,
        help="Max image dimension sebelum inference (default=1024, 0=no resize)"
    )
    args = parser.parse_args()

    if args.file:
        console.print(Panel(f"[bold]Testing:[/bold] {args.file.name}", expand=False))
        if not check_health(args.url):
            sys.exit(1)
        out = predict(args.file, args.url, preprocess=args.preprocess, max_dim=args.max_dim)
        if out:
            _offer_open(out)
    else:
        interactive_mode(args.url)


if __name__ == "__main__":
    main()
