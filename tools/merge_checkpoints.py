#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.checkpoint import ALPHA_KEY, alpha_keys, checkpoint_state


def load_payload(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Merge an MVSFormer++ checkpoint with trained AlphaNet weights"
        )
    )
    parser.add_argument("--mvs_checkpoint", type=Path, required=True)
    parser.add_argument("--alpha_checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.output.exists() and not args.force:
        raise FileExistsError(
            f"{args.output} exists; pass --force to replace it"
        )

    base_payload = load_payload(args.mvs_checkpoint)
    base_state = checkpoint_state(base_payload)
    alpha_state = alpha_keys(
        checkpoint_state(load_payload(args.alpha_checkpoint)),
        stages=(0, 1, 2, 3),
    )
    present_stages = {
        int(ALPHA_KEY.match(key).group(1)) for key in alpha_state
    }
    if not {0, 1, 2}.issubset(present_stages):
        raise ValueError(
            "Alpha checkpoint must contain stages 1-3; found "
            + str(sorted(stage + 1 for stage in present_stages))
        )

    base_state.update(alpha_state)
    if isinstance(base_payload, dict) and any(
        key in base_payload for key in ("state_dict", "model")
    ):
        merged = dict(base_payload)
        merged.pop("model", None)
        merged["state_dict"] = base_state
    else:
        merged = {"state_dict": base_state}
    merged["mr_mvs_merge"] = {
        "mvs_checkpoint": str(args.mvs_checkpoint),
        "alpha_checkpoint": str(args.alpha_checkpoint),
        "alpha_tensors": len(alpha_state),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(merged, args.output)
    print(
        f"Saved {args.output}: {len(base_state)} total tensors, "
        f"{len(alpha_state)} AlphaNet tensors"
    )


if __name__ == "__main__":
    main()
