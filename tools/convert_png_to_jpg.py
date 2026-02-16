from __future__ import annotations

import os
import subprocess
from pathlib import Path


FFMPEG = r"C:\Users\zhou\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.0.1-full_build\bin\ffmpeg.exe"

MAX_DIM = 1200  # max width/height after resize
Q = 3  # ffmpeg jpeg quality scale (lower is better): 2..5 is typical


def run_ffmpeg(in_path: Path, out_path: Path) -> bool:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    vf = f"scale={MAX_DIM}:{MAX_DIM}:force_original_aspect_ratio=decrease"
    cmd = [
        FFMPEG,
        "-y",
        "-v",
        "error",
        "-i",
        str(in_path),
        "-vf",
        vf,
        "-q:v",
        str(Q),
        "-frames:v",
        "1",
        str(out_path),
    ]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        print("[fail]", in_path, "->", out_path)
        if p.stderr.strip():
            print(p.stderr.strip())
        return False
    return out_path.exists() and out_path.stat().st_size > 0


def convert_tree(root: Path) -> tuple[int, int]:
    converted = 0
    failed = 0
    for png in sorted(root.rglob("*.png")):
        if not png.is_file():
            continue
        jpg = png.with_suffix(".jpg")
        if jpg.exists() and jpg.stat().st_size > 0:
            # If jpg already exists, drop png to reduce size.
            try:
                png.unlink()
            except OSError:
                pass
            continue
        ok = run_ffmpeg(png, jpg)
        if ok:
            converted += 1
            try:
                png.unlink()
            except OSError:
                pass
        else:
            failed += 1
    return converted, failed


def main():
    targets = [Path("static/render_compare"), Path("static/image")]
    total_c = 0
    total_f = 0
    for t in targets:
        if not t.exists():
            continue
        c, f = convert_tree(t)
        total_c += c
        total_f += f
        print("[done]", t, "converted", c, "failed", f)
    print("total_converted", total_c, "total_failed", total_f)


if __name__ == "__main__":
    main()

