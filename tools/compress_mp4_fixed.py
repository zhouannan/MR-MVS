from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path


FFMPEG = r"C:\Users\zhou\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.0.1-full_build\bin\ffmpeg.exe"

IN_DIR = Path("static/mp4/fixed")
MAX_DIM = 960


def compress_one(src: Path) -> bool:
    # Render videos are visually smooth but big; use higher CRF.
    is_render = "render_" in src.name
    crf = "32" if is_render else "30"
    preset = "slow"

    tmp = src.with_suffix(".tmp.mp4")
    bak = src.with_suffix(".bak.mp4")

    vf = f"scale={MAX_DIM}:{MAX_DIM}:force_original_aspect_ratio=decrease"
    cmd = [
        FFMPEG,
        "-y",
        "-v",
        "error",
        "-i",
        str(src),
        "-vf",
        vf,
        "-c:v",
        "libx264",
        "-crf",
        crf,
        "-preset",
        preset,
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        "-an",
        str(tmp),
    ]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0 or not tmp.exists() or tmp.stat().st_size == 0:
        if p.stderr.strip():
            print("[fail]", src.name, p.stderr.strip())
        if tmp.exists():
            tmp.unlink()
        return False

    # Replace in-place with a backup for safety during this run.
    if bak.exists():
        bak.unlink()
    src.replace(bak)
    tmp.replace(src)
    bak.unlink(missing_ok=True)  # keep repo clean
    return True


def main():
    if not IN_DIR.exists():
        print("No static/mp4/fixed directory.")
        return
    mp4s = sorted([p for p in IN_DIR.glob("*.mp4") if p.is_file()])
    ok = 0
    fail = 0
    for p in mp4s:
        before = p.stat().st_size
        if compress_one(p):
            after = p.stat().st_size
            ok += 1
            print("[ok]", p.name, before, "->", after)
        else:
            fail += 1
    print("done ok=", ok, "fail=", fail)


if __name__ == "__main__":
    main()

