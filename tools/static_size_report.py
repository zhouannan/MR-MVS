from __future__ import annotations

from pathlib import Path


def walk_size(root: Path):
    total = 0
    files = []
    for p in root.rglob("*"):
        if p.is_file():
            sz = p.stat().st_size
            total += sz
            files.append((sz, p))
    files.sort(reverse=True, key=lambda x: x[0])
    return total, files


def main():
    static = Path("static")
    total, files = walk_size(static)
    pngs = list((static / "render_compare").rglob("*.png")) if (static / "render_compare").exists() else []
    jpgs = list((static / "render_compare").rglob("*.jpg")) if (static / "render_compare").exists() else []
    mp4s = list(static.rglob("*.mp4")) if static.exists() else []

    print("static_total_bytes", total)
    print("render_compare_png_count", len([p for p in pngs if p.is_file()]))
    print("render_compare_jpg_count", len([p for p in jpgs if p.is_file()]))
    print("mp4_count", len([p for p in mp4s if p.is_file()]))
    print("top10_files:")
    for sz, p in files[:10]:
        print(sz, str(p).replace("\\", "/"))


if __name__ == "__main__":
    main()

