import argparse
from pathlib import Path

import cv2
import numpy as np
from plyfile import PlyData, PlyElement


def resize_depth_map(depth_map, intrinsics, image_height, image_width):
    """Scale intrinsics from an image grid to a depth-map grid."""
    depth_height, depth_width = depth_map.shape
    scaled = intrinsics.copy()
    scaled[0] *= depth_width / float(image_width)
    scaled[1] *= depth_height / float(image_height)
    return scaled


def read_cam_file(filename):
    with open(filename, "r", encoding="utf-8") as handle:
        lines = [line.rstrip() for line in handle]
    extrinsics = np.fromstring(
        " ".join(lines[1:5]), dtype=np.float32, sep=" "
    ).reshape(4, 4)
    intrinsics = np.fromstring(
        " ".join(lines[7:10]), dtype=np.float32, sep=" "
    ).reshape(3, 3)
    return intrinsics, extrinsics


def depth_to_point_cloud(depth_map, intrinsics, extrinsics):
    """Back-project positive camera depths into world coordinates."""
    height, width = depth_map.shape
    pixel_x, pixel_y = np.meshgrid(
        np.arange(width, dtype=np.float32),
        np.arange(height, dtype=np.float32),
    )
    valid = np.isfinite(depth_map) & (depth_map > 0)
    depth = depth_map[valid]

    camera_points = np.stack(
        [
            (pixel_x[valid] - intrinsics[0, 2])
            / intrinsics[0, 0]
            * depth,
            (pixel_y[valid] - intrinsics[1, 2])
            / intrinsics[1, 1]
            * depth,
            depth,
            np.ones_like(depth),
        ],
        axis=1,
    )
    camera_to_world = np.linalg.inv(extrinsics)
    return (camera_to_world @ camera_points.T).T[:, :3], valid


def save_ply(filename, coordinates, colors):
    vertices = np.empty(
        len(coordinates),
        dtype=[
            ("x", "f4"),
            ("y", "f4"),
            ("z", "f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
        ],
    )
    for index, name in enumerate(("x", "y", "z")):
        vertices[name] = coordinates[:, index]
    for index, name in enumerate(("red", "green", "blue")):
        vertices[name] = colors[:, index]
    PlyData(
        [PlyElement.describe(vertices, "vertex")],
        text=False,
    ).write(filename)


def convert_directory(
    camera_dir,
    image_dir,
    depth_dir,
    output_dir,
    *,
    allow_camera_rescale=False,
):
    camera_dir = Path(camera_dir)
    image_dir = Path(image_dir)
    depth_dir = Path(depth_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    depth_paths = sorted(depth_dir.glob("*.npy"))
    if not depth_paths:
        raise FileNotFoundError(f"No NPY depths found in {depth_dir}")

    for depth_path in depth_paths:
        stem = depth_path.stem
        image_path = image_dir / f"{stem}.jpg"
        camera_path = camera_dir / f"{stem}_cam.txt"
        image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise FileNotFoundError(image_path)

        depth = np.load(depth_path).astype(np.float32)
        intrinsics, extrinsics = read_cam_file(camera_path)
        image_height, image_width = image_bgr.shape[:2]
        if depth.shape != (image_height, image_width):
            if not allow_camera_rescale:
                raise ValueError(
                    f"{stem}: image {image_bgr.shape[:2]} does not match "
                    f"depth {depth.shape}"
                )
            intrinsics = resize_depth_map(
                depth,
                intrinsics,
                image_height,
                image_width,
            )
            image_bgr = cv2.resize(
                image_bgr,
                (depth.shape[1], depth.shape[0]),
                interpolation=cv2.INTER_LINEAR,
            )

        coordinates, valid = depth_to_point_cloud(
            depth,
            intrinsics,
            extrinsics,
        )
        colors = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)[valid]
        output_path = output_dir / f"{stem}.ply"
        save_ply(output_path, coordinates, colors)
        print(f"Saved {output_path}: {len(coordinates)} points")


def main():
    parser = argparse.ArgumentParser(
        description="Back-project MR-MVS NPY depth maps into per-view PLY files"
    )
    parser.add_argument("--cams_path", required=True)
    parser.add_argument("--image_path", required=True)
    parser.add_argument("--depth_path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--allow_camera_rescale", action="store_true")
    args = parser.parse_args()
    convert_directory(
        args.cams_path,
        args.image_path,
        args.depth_path,
        args.output,
        allow_camera_rescale=args.allow_camera_rescale,
    )


if __name__ == "__main__":
    main()
