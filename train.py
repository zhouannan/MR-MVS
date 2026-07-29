import argparse
import os

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from base.parse_config import ConfigParser
from trainer.mvsformer_trainer import Trainer
from utils import (
    get_lr_schedule_with_warmup,
    get_parameter_groups,
    init_model,
)


SEED = 123
torch.manual_seed(SEED)
cudnn.benchmark = True
cudnn.deterministic = False


def _dataset_class(loader_type, multi_scale):
    if loader_type == "BlendedLoader":
        if multi_scale:
            from datasets.blended_dataset_ms import BlendedMVSDataset
        else:
            from datasets.blended_dataset import BlendedMVSDataset
        return BlendedMVSDataset
    if loader_type == "DTULoader":
        if multi_scale:
            from datasets.dtu_dataset_ms import DTUMVSDataset
        else:
            from datasets.dtu_dataset import DTUMVSDataset
        return DTUMVSDataset
    raise ValueError(f"Unsupported data loader: {loader_type}")


def _make_dataset_args(raw_args, *, mode, listfile, world_size):
    args = dict(raw_args)
    args.pop("train_data_list", None)
    args.pop("val_data_list", None)
    args.pop("num_workers", None)
    args.pop("shuffle", None)
    args.pop("data_set_type", None)
    args["listfile"] = listfile
    args["mode"] = mode
    args["world_size"] = world_size
    if "num_depths" in args:
        args["ndepths"] = args.pop("num_depths")
    return args


def _load_model_state(model, path, *, strict):
    checkpoint = torch.load(path, map_location="cpu")
    raw_state = checkpoint.get(
        "state_dict", checkpoint.get("model", checkpoint)
    )
    state = {}
    for key, value in raw_state.items():
        key = key[7:] if key.startswith("module.") else key
        if "pe_dict" not in key:
            state[key] = value
    incompatible = model.load_state_dict(state, strict=strict)
    if not strict:
        print(f"Loaded initialization checkpoint: {path}")
        print(f"  missing keys: {len(incompatible.missing_keys)}")
        print(f"  unexpected keys: {len(incompatible.unexpected_keys)}")
    return checkpoint


def _build_loaders(config, rank, world_size):
    if len(config["data_loader"]) != 1:
        raise ValueError(
            "MR-MVS uses one training dataset per run; "
            "put additional datasets in separate configs."
        )
    loader_config = config["data_loader"][0]
    loader_type = loader_config["type"]
    raw_args = dict(loader_config["args"])
    multi_scale = bool(raw_args.get("multi_scale", False))
    if multi_scale:
        cudnn.benchmark = False
    dataset_class = _dataset_class(loader_type, multi_scale)

    batch_size = int(raw_args["batch_size"]) // world_size
    if batch_size < 1:
        raise ValueError("Per-process batch size must be at least one")
    num_workers = int(raw_args.get("num_workers", 4))

    train_args = _make_dataset_args(
        raw_args,
        mode="train",
        listfile=raw_args["train_data_list"],
        world_size=world_size,
    )
    train_args["batch_size"] = batch_size
    train_dataset = dataset_class(**train_args)
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_args = _make_dataset_args(
        raw_args,
        mode="val",
        listfile=raw_args["val_data_list"],
        world_size=world_size,
    )
    val_args.update(
        {
            "nviews": int(raw_args.get("val_nviews", 5)),
            "batch_size": 1,
            "random_crop": False,
            "augment": False,
        }
    )
    val_dataset = dataset_class(**val_args)
    val_sampler = DistributedSampler(
        val_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        sampler=val_sampler,
        shuffle=False,
        num_workers=min(num_workers, 4),
        pin_memory=True,
    )
    return [train_loader], [val_loader], train_sampler


def main_worker(local_rank, args, config):
    rank = args.node_rank * args.gpus_per_node + local_rank
    torch.cuda.set_device(local_rank)
    if args.ddp:
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            world_size=args.world_size,
            rank=rank,
        )
        print(
            f"DDP node={args.node_rank}/{args.nodes} "
            f"rank={rank}/{args.world_size} gpu={local_rank}"
        )

    train_loaders, val_loaders, train_sampler = _build_loaders(
        config, rank, args.world_size
    )
    config["arch"]["dataset_name"] = config["data_loader"][0]["type"]
    model = init_model(config)

    if not config["arch"]["args"].get("freeze_vit", True):
        for name, parameter in model.named_parameters():
            if name.startswith("vit."):
                parameter.requires_grad = name != "vit.mask_token"

    optimizer_args = config["optimizer"]["args"]
    parameter_groups = get_parameter_groups(
        optimizer_args,
        model,
        freeze_vit=config["arch"]["args"].get("freeze_vit", True),
    )
    optimizer = torch.optim.AdamW(
        parameter_groups,
        lr=optimizer_args["lr"],
        weight_decay=optimizer_args["weight_decay"],
    )
    total_steps = (
        len(train_loaders[0]) * config["trainer"]["epochs"]
    )
    scheduler = get_lr_schedule_with_warmup(
        optimizer,
        num_warmup_steps=optimizer_args["warmup_steps"],
        min_lr=optimizer_args["min_lr"],
        total_steps=total_steps,
    )

    start_epoch = 1
    if args.pretrained:
        _load_model_state(model, args.pretrained, strict=False)
    if args.resume:
        checkpoint = _load_model_state(model, args.resume, strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = int(checkpoint["epoch"]) + 1
        for _ in range(int(checkpoint["epoch"]) * len(train_loaders[0])):
            scheduler.step()
        print(f"Resuming at epoch {start_epoch}")

    model = model.cuda(local_rank)
    if args.ddp:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False,
        )

    writer = SummaryWriter(config.log_dir) if rank == 0 else None
    trainer = Trainer(
        model,
        optimizer,
        config=config,
        data_loader=train_loaders,
        valid_data_loader=val_loaders,
        lr_scheduler=scheduler,
        writer=writer,
        rank=rank,
        ddp=args.ddp,
        train_sampler=train_sampler,
        debug=args.debug,
    )
    trainer.start_epoch = start_epoch
    trainer.train()

    if writer is not None:
        writer.close()
    if args.ddp:
        dist.destroy_process_group()


def build_parser():
    parser = argparse.ArgumentParser(
        description="Train MR-MVS AlphaNet on MVSFormer++"
    )
    parser.add_argument("-c", "--config", required=True)
    parser.add_argument("-e", "--exp_name", default=None)
    parser.add_argument("-r", "--resume", default=None)
    parser.add_argument(
        "--pretrained",
        default=None,
        help="MVSFormer++ or merged checkpoint used for initialization",
    )
    parser.add_argument("-d", "--device", default=None)
    parser.add_argument("--data_path", default=None)
    parser.add_argument("--mono_depths_path", default=None)
    parser.add_argument("--nodes", type=int, default=1)
    parser.add_argument("--node_rank", type=int, default=0)
    parser.add_argument("--master_addr", default="127.0.0.1")
    parser.add_argument("--master_port", default="1122")
    parser.add_argument("--ddp", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--batch_size", "--bs", type=int, default=None)
    return parser


if __name__ == "__main__":
    parser = build_parser()
    config = ConfigParser.from_args(parser, options=[])
    args = parser.parse_args()

    if args.lr is not None:
        config["optimizer"]["args"]["lr"] = args.lr
    if args.batch_size is not None:
        config["data_loader"][0]["args"]["batch_size"] = args.batch_size
    if args.data_path:
        config["data_loader"][0]["args"]["datapath"] = args.data_path
    if args.mono_depths_path:
        config["data_loader"][0]["args"][
            "mono_depths_path"
        ] = args.mono_depths_path

    args.gpus_per_node = torch.cuda.device_count()
    if args.gpus_per_node < 1:
        raise RuntimeError("MR-MVS training requires at least one CUDA GPU")
    args.world_size = (
        args.nodes * args.gpus_per_node if args.ddp else 1
    )
    os.environ["MASTER_ADDR"] = args.master_addr
    os.environ["MASTER_PORT"] = args.master_port

    if args.ddp:
        mp.spawn(
            main_worker,
            nprocs=args.gpus_per_node,
            args=(args, config),
        )
    else:
        main_worker(0, args, config)
