import re

import torch


ALPHA_KEY = re.compile(r"^fusions\.(\d+)\.alpha_net\.")


def checkpoint_state(checkpoint):
    if isinstance(checkpoint, str):
        checkpoint = torch.load(checkpoint, map_location="cpu")
    state = checkpoint.get(
        "state_dict", checkpoint.get("model", checkpoint)
    )
    result = {}
    for key, value in state.items():
        key = key[7:] if key.startswith("module.") else key
        if "pe_dict" not in key:
            result[key] = value
    return result


def alpha_keys(state, stages=(0, 1, 2)):
    stages = set(int(stage) for stage in stages)
    return {
        key: value
        for key, value in state.items()
        if (match := ALPHA_KEY.match(key))
        and int(match.group(1)) in stages
    }


def load_model_checkpoint(
    model,
    checkpoint_path,
    *,
    alpha_checkpoint_path=None,
    require_alpha=False,
):
    state = checkpoint_state(checkpoint_path)
    if alpha_checkpoint_path:
        alpha_state = alpha_keys(
            checkpoint_state(alpha_checkpoint_path),
            stages=(0, 1, 2, 3),
        )
        if not alpha_state:
            raise ValueError(
                f"No AlphaNet tensors found in {alpha_checkpoint_path}"
            )
        state.update(alpha_state)

    present_alpha = alpha_keys(state)
    if require_alpha:
        present_stages = {
            int(ALPHA_KEY.match(key).group(1))
            for key in present_alpha
        }
        if present_stages != {0, 1, 2}:
            raise ValueError(
                "Monocular inference requires trained AlphaNet weights for "
                f"stages 1-3; found stages {sorted(stage + 1 for stage in present_stages)}"
            )

    incompatible = model.load_state_dict(state, strict=False)
    unexpected = list(incompatible.unexpected_keys)
    missing = list(incompatible.missing_keys)
    non_alpha_missing = [
        key for key in missing if ".alpha_net." not in key
    ]
    if non_alpha_missing:
        raise RuntimeError(
            "Checkpoint is missing non-AlphaNet model tensors: "
            + ", ".join(non_alpha_missing[:20])
        )
    print(f"Loaded checkpoint: {checkpoint_path}")
    if alpha_checkpoint_path:
        print(f"Loaded AlphaNet override: {alpha_checkpoint_path}")
    print(
        f"Checkpoint report: missing={len(missing)}, "
        f"unexpected={len(unexpected)}, alpha_tensors={len(present_alpha)}"
    )
    if unexpected:
        print("Ignored unexpected keys:", unexpected[:20])
    return {
        "missing_keys": missing,
        "unexpected_keys": unexpected,
        "alpha_tensors": len(present_alpha),
    }
