#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def read_camera(path):
    lines = path.read_text(encoding="utf-8").splitlines()
    extrinsics = np.fromstring(
        " ".join(lines[1:5]), sep=" ", dtype=np.float32
    ).reshape(4, 4)
    intrinsics = np.fromstring(
        " ".join(lines[7:10]), sep=" ", dtype=np.float32
    ).reshape(3, 3)
    depth_parameters = np.fromstring(
        lines[11], sep=" ", dtype=np.float32
    )
    return intrinsics, extrinsics, depth_parameters


def main():
    parser = argparse.ArgumentParser(
        description="Validate an MR-MVS scene output before fusion"
    )
    parser.add_argument("--scene_dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    depth_paths = sorted((args.scene_dir / "depth_est").glob("*.npy"))
    if args.limit > 0:
        depth_paths = depth_paths[: args.limit]
    if not depth_paths:
        raise FileNotFoundError("No depth_est/*.npy files found")
    if not (args.scene_dir / "pair.txt").is_file():
        raise FileNotFoundError("pair.txt is missing")

    valid_depth_values = 0
    for depth_path in depth_paths:
        stem = depth_path.stem
        depth = np.load(depth_path)
        confidence = np.load(
            args.scene_dir / "confidence" / f"{stem}.npy"
        )
        stage_confidence = np.load(
            args.scene_dir / "stage_4_confidence" / f"{stem}.npy"
        )
        image_path = args.scene_dir / "images_test" / f"{stem}.jpg"
        camera_path = (
            args.scene_dir / "cams_test" / f"{stem}_cam.txt"
        )
        with Image.open(image_path) as image:
            image_hw = (image.height, image.width)
        if depth.shape != image_hw:
            raise ValueError(
                f"{stem}: depth {depth.shape} != image {image_hw}"
            )
        if confidence.shape != depth.shape:
            raise ValueError(f"{stem}: confidence shape mismatch")
        if stage_confidence.shape != depth.shape:
            raise ValueError(f"{stem}: stage confidence shape mismatch")
        if not np.isfinite(depth).all():
            raise ValueError(f"{stem}: depth contains NaN or Inf")

        intrinsics, extrinsics, depth_parameters = read_camera(
            camera_path
        )
        if not np.isfinite(intrinsics).all():
            raise ValueError(f"{stem}: intrinsics contain NaN or Inf")
        if not np.isfinite(extrinsics).all():
            raise ValueError(f"{stem}: extrinsics contain NaN or Inf")
        if intrinsics[0, 0] <= 0 or intrinsics[1, 1] <= 0:
            raise ValueError(f"{stem}: non-positive focal length")
        if depth_parameters.size != 4:
            raise ValueError(f"{stem}: invalid depth-range camera row")
        if depth_parameters[1] <= 0 or depth_parameters[2] <= 0:
            raise ValueError(f"{stem}: invalid depth hypotheses")
        valid_depth_values += int(np.count_nonzero(depth > 0))

    first_shape = np.load(depth_paths[0], mmap_mode="r").shape
    print(
        f"validated={len(depth_paths)} "
        f"resolution={first_shape} "
        f"positive_depth_values={valid_depth_values}"
    )
    print("pixel grid and camera metadata: OK")


if __name__ == "__main__":
    main()
