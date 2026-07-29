import argparse
import json
import os
import time

import numpy as np
import torch

from datasets.mvs_dataset_all import MVSDatasetAll
from models.checkpoint import load_model_checkpoint
from models.multi_stage_infer import MultiStageInferencer
from test import (
    amp_dtype,
    copy_pair_file,
    load_config,
    parse_tmp,
    save_prediction,
)
from utils import init_model


def write_run_config(args):
    os.makedirs(args.outdir, exist_ok=True)
    path = os.path.join(args.outdir, "mr_mvs_run.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(vars(args), handle, indent=2, sort_keys=True)


@torch.no_grad()
def run(args):
    config = load_config(args.config)
    model_args = config["arch"]["args"]
    model_args.update(
        {
            "stage_refine_iters": args.refine_iters,
            "stage_refine_reproj_threshold": args.reproj_threshold,
            "stage_refine_use_noise": int(args.refine_noise),
            "stage_refine_max_srcs": args.refine_max_srcs,
        }
    )
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

    dataset = MVSDatasetAll(
        datapath=args.testpath,
        listfile=args.testlist,
        mode="multi_stage",
        nviews=args.num_view,
        mono_depths_path=args.mono_depths_path,
        ndepths=args.numdepth,
        interval_scale=args.interval_scale,
        max_h=args.max_h,
        max_w=args.max_w,
        dataset=args.dataset,
        chunk_size=args.chunk_size,
        view_cache_size=args.view_cache_size,
        use_short_range=args.use_short_range,
    )
    dtype = amp_dtype(args.amp_dtype)
    inferencer = MultiStageInferencer(
        model=model,
        device=device,
        num_stages=4,
        tmp=parse_tmp(args.tmps),
        enable_pair_cache=args.pair_cache_size > 0,
        max_pair_cache_size=args.pair_cache_size,
        profile_timing=args.profile_timing,
        amp_dtype=dtype,
    )
    write_run_config(args)

    total_chunks = len(dataset)
    if args.max_batches is not None:
        total_chunks = min(total_chunks, max(0, args.max_batches))
    saved = 0
    wall_start = time.time()
    last_scan = None

    for chunk_index in range(total_chunks):
        sample = dataset[chunk_index]
        scan = sample["scan"]
        copy_pair_file(args.testpath, scan, args.outdir)
        if last_scan is not None and scan != last_scan:
            inferencer.clear_cache()
        last_scan = scan

        inferencer.reset_timing()
        results = inferencer.run(
            imgs_cpu=sample["imgs_all"],
            projections=sample["projs_all"],
            mono_depths=sample.get("mono_all"),
            depth_values=sample["depth_values_all"],
            pairs=sample["pairs"],
            view_ids=sample["view_ids"],
            view_num=sample["view_num"],
            cache_namespace=scan,
        )
        view_local = {
            int(view_id): index
            for index, view_id in enumerate(sample["view_ids"])
        }
        for view_id, result in results.items():
            local_index = view_local[int(view_id)]
            save_prediction(
                args.outdir,
                scan,
                int(view_id),
                result["depth"].squeeze(0).numpy(),
                result["confidence"].squeeze(0).numpy(),
                result["stage_4_confidence"].squeeze(0).numpy(),
                sample["imgs_all"][0, local_index],
                sample["projs_all"][local_index]["stage4"][0, 0],
            )
            saved += 1

        message = (
            f"[chunk {chunk_index + 1}/{total_chunks}] "
            f"scene={scan} saved={len(results)} total={saved}"
        )
        if args.profile_timing:
            timing = inferencer.timing_summary(len(results))
            message += (
                f" forward={timing['forward_s']:.2f}s"
                f" refine={timing['refine_s']:.2f}s"
            )
        print(message, flush=True)

    print(
        f"Saved {saved} refined depth maps to {args.outdir} "
        f"in {(time.time() - wall_start) / 60.0:.1f} min"
    )


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Run MR-MVS AlphaNet inference and final-stage geometric refinement"
        )
    )
    parser.add_argument("--config", default="config/mr_mvs_tnt.json")
    parser.add_argument("--model", required=True)
    parser.add_argument("--alpha_checkpoint", default=None)
    parser.add_argument("--vit_path", default=None)
    parser.add_argument("--testpath", required=True)
    parser.add_argument("--testlist", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--mono_depths_path", default=None)
    parser.add_argument(
        "--dataset", choices=("tt", "dtu", "eth3d"), default="tt"
    )
    parser.add_argument("--num_view", type=int, default=5)
    parser.add_argument("--numdepth", type=int, default=192)
    parser.add_argument("--interval_scale", type=float, default=1.06)
    parser.add_argument("--max_h", type=int, default=1088)
    parser.add_argument("--max_w", type=int, default=1920)
    parser.add_argument("--tmps", default="5,5,5,1")
    parser.add_argument(
        "--amp_dtype", choices=("none", "fp16", "bf16"), default="bf16"
    )
    parser.add_argument("--alpha_scale", type=float, default=None)
    parser.add_argument("--refine_iters", type=int, default=10)
    parser.add_argument(
        "--reproj_threshold",
        type=float,
        default=0.01,
        help=(
            "Pixel reprojection threshold. Each source is rejected "
            "independently when its error exceeds this value."
        ),
    )
    parser.add_argument(
        "--refine_noise",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--refine_max_srcs", type=int, default=0)
    parser.add_argument("--chunk_size", type=int, default=50)
    parser.add_argument("--view_cache_size", type=int, default=64)
    parser.add_argument("--pair_cache_size", type=int, default=128)
    parser.add_argument("--profile_timing", action="store_true")
    parser.add_argument("--max_batches", type=int, default=None)
    parser.add_argument("--use_short_range", action="store_true")
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
