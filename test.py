import argparse
import json
import os
import shutil
import time

import cv2
import numpy as np
import torch

from datasets.data_loaders import DTULoader
from datasets.data_io import write_cam
from models.checkpoint import load_model_checkpoint
from utils import init_model


MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def load_config(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def parse_tmp(value):
    result = [float(item) for item in value.split(",")]
    if len(result) != 4:
        raise ValueError("--tmps must contain four comma-separated values")
    return result


def amp_dtype(name):
    if name == "none":
        return None
    if name == "fp16":
        return torch.float16
    if name == "bf16":
        return torch.bfloat16
    raise ValueError(name)


def save_image_and_camera(scene_root, view_id, image, camera):
    image = image.detach().cpu() * STD + MEAN
    image = (
        image.clamp(0, 1).permute(1, 2, 0).numpy() * 255
    ).astype(np.uint8)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    camera = camera.detach().cpu().numpy()

    for image_dir, camera_dir in (
        ("images", "cams"),
        ("images_test", "cams_test"),
    ):
        image_path = os.path.join(
            scene_root, image_dir, f"{view_id:08d}.jpg"
        )
        camera_path = os.path.join(
            scene_root, camera_dir, f"{view_id:08d}_cam.txt"
        )
        os.makedirs(os.path.dirname(image_path), exist_ok=True)
        os.makedirs(os.path.dirname(camera_path), exist_ok=True)
        cv2.imwrite(image_path, image)
        write_cam(camera_path, camera)


def save_prediction(
    outdir,
    scan,
    view_id,
    depth,
    confidence,
    stage_confidence,
    image,
    camera,
):
    scene_root = os.path.join(outdir, scan)
    outputs = {
        "depth_est": depth.astype(np.float32),
        "confidence": np.clip(confidence * 255, 0, 255).astype(np.uint8),
        "stage_4_confidence": np.clip(
            stage_confidence * 255, 0, 255
        ).astype(np.uint8),
    }
    for directory, array in outputs.items():
        path = os.path.join(
            scene_root, directory, f"{view_id:08d}.npy"
        )
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.save(path, array)
    save_image_and_camera(scene_root, view_id, image, camera)


def copy_pair_file(testpath, scan, outdir):
    destination = os.path.join(outdir, scan, "pair.txt")
    if os.path.isfile(destination):
        return
    for name in ("pair.txt", "new_pair.txt"):
        source = os.path.join(testpath, scan, name)
        if os.path.isfile(source):
            os.makedirs(os.path.dirname(destination), exist_ok=True)
            shutil.copyfile(source, destination)
            return
    raise FileNotFoundError(f"No pair file found for scene {scan}")


@torch.no_grad()
def run(args):
    config = load_config(args.config)
    model_args = config["arch"]["args"]
    model_args["stage_refine_iters"] = 0
    if args.vit_path:
        model_args["vit_path"] = args.vit_path
    if args.alpha_scale is not None:
        model_args.setdefault("alpha_fusion", {})[
            "alpha_scale"
        ] = args.alpha_scale

    model = init_model(config)
    load_model_checkpoint(
        model,
        args.model,
        alpha_checkpoint_path=args.alpha_checkpoint,
        require_alpha=args.mono_depths_path is not None,
    )
    device = torch.device("cuda")
    model = model.to(device).eval()

    loader = DTULoader(
        data_path=args.testpath,
        data_list=args.testlist,
        mode="test",
        num_srcs=args.num_view,
        mono_depths_path=args.mono_depths_path,
        num_depths=args.numdepth,
        interval_scale=args.interval_scale,
        shuffle=False,
        batch_size=1,
        fix_res=True,
        max_h=args.max_h,
        max_w=args.max_w,
        dataset_eval=args.dataset,
        num_workers=args.num_workers,
        use_short_range=args.use_short_range,
    )

    dtype = amp_dtype(args.amp_dtype)
    temperatures = parse_tmp(args.tmps)
    total = len(loader)
    if args.max_batches is not None:
        total = min(total, max(0, args.max_batches))

    elapsed = []
    for batch_index, sample in enumerate(loader):
        if batch_index >= total:
            break
        images = sample["imgs"].to(device, non_blocking=True)
        cameras = {
            stage: value.to(device, non_blocking=True)
            for stage, value in sample["proj_matrices"].items()
        }
        depth_values = sample["depth_values"].to(
            device, non_blocking=True
        )
        mono_depth = None
        if "mono_depth" in sample:
            mono_depth = {
                stage: value.to(device, non_blocking=True)
                for stage, value in sample["mono_depth"].items()
            }

        torch.cuda.synchronize()
        start = time.time()
        with torch.autocast(
            device_type="cuda",
            dtype=dtype or torch.float32,
            enabled=dtype is not None,
        ):
            outputs = model(
                images,
                cameras,
                depth_values,
                mono_depth=mono_depth,
                tmp=temperatures,
            )
        torch.cuda.synchronize()
        elapsed.append(time.time() - start)

        depths = outputs["refined_depth"].float().cpu().numpy()
        confidences = (
            outputs["photometric_confidence"].float().cpu().numpy()
        )
        stage_confidences = (
            outputs["stage4"]["photometric_confidence"]
            .float()
            .cpu()
            .numpy()
        )
        stage4_cameras = sample["proj_matrices"]["stage4"]

        for item_index in range(depths.shape[0]):
            filename = sample["filename"][item_index]
            scan = filename.split("/")[0]
            copy_pair_file(args.testpath, scan, args.outdir)
            view_id = int(
                filename.rsplit("/", 1)[-1].split("{")[0]
            )
            save_prediction(
                args.outdir,
                scan,
                view_id,
                depths[item_index],
                confidences[item_index],
                stage_confidences[item_index],
                sample["imgs"][item_index, 0],
                stage4_cameras[item_index, 0],
            )
        print(
            f"[{batch_index + 1}/{total}] "
            f"{sample['filename'][0]} "
            f"{elapsed[-1]:.3f}s "
            f"{depths.shape[-2]}x{depths.shape[-1]}",
            flush=True,
        )

    if elapsed:
        print(
            f"Saved {len(elapsed)} depth maps to {args.outdir}; "
            f"mean inference time {np.mean(elapsed):.3f}s"
        )


def build_parser():
    parser = argparse.ArgumentParser(
        description="Run MVSFormer++ or AlphaNet-guided MR-MVS inference"
    )
    parser.add_argument("--config", default="config/mr_mvs_tnt.json")
    parser.add_argument("--model", required=True)
    parser.add_argument("--alpha_checkpoint", default=None)
    parser.add_argument("--vit_path", default=None)
    parser.add_argument("--testpath", required=True)
    parser.add_argument("--testlist", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument(
        "--mono_depths_path",
        default=None,
        help="Omit this argument to run the MVSFormer++ baseline",
    )
    parser.add_argument(
        "--dataset", choices=("tt", "dtu", "eth3d"), default="tt"
    )
    parser.add_argument("--num_view", type=int, default=5)
    parser.add_argument("--numdepth", type=int, default=192)
    parser.add_argument("--interval_scale", type=float, default=1.06)
    parser.add_argument("--max_h", type=int, default=1088)
    parser.add_argument("--max_w", type=int, default=1920)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--tmps", default="5,5,5,1")
    parser.add_argument(
        "--amp_dtype", choices=("none", "fp16", "bf16"), default="bf16"
    )
    parser.add_argument("--alpha_scale", type=float, default=None)
    parser.add_argument("--max_batches", type=int, default=None)
    parser.add_argument("--use_short_range", action="store_true")
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
