#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.checkpoint import load_model_checkpoint
from utils import init_model


def main():
    parser = argparse.ArgumentParser(
        description="Construct MR-MVS and validate checkpoint coverage"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--alpha_checkpoint", type=Path, default=None)
    parser.add_argument("--vit_path", type=Path, default=None)
    parser.add_argument("--require_alpha", action="store_true")
    args = parser.parse_args()

    with args.config.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    if args.vit_path:
        config["arch"]["args"]["vit_path"] = str(args.vit_path)
    model = init_model(config)
    report = load_model_checkpoint(
        model,
        str(args.checkpoint),
        alpha_checkpoint_path=(
            str(args.alpha_checkpoint) if args.alpha_checkpoint else None
        ),
        require_alpha=args.require_alpha,
    )
    print(f"parameters={sum(p.numel() for p in model.parameters())}")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
