import argparse

import numpy as np


def main():
    parser = argparse.ArgumentParser(
        description="Compare two MR-MVS NPY depth maps."
    )
    parser.add_argument("first")
    parser.add_argument("second")
    parser.add_argument("--epsilon", type=float, default=1e-6)
    args = parser.parse_args()

    first = np.load(args.first).astype(np.float32)
    second = np.load(args.second).astype(np.float32)
    if first.shape != second.shape:
        raise ValueError(
            f"Depth shapes differ: {first.shape} versus {second.shape}"
        )

    valid = (
        np.isfinite(first)
        & np.isfinite(second)
        & (first > 0)
        & (second > 0)
    )
    if not np.any(valid):
        raise ValueError("No jointly valid positive depth pixels")

    difference = np.abs(first[valid] - second[valid])
    print(f"valid_pixels={difference.size}")
    print(f"mean_abs_difference={difference.mean():.9f}")
    print(f"max_abs_difference={difference.max():.9f}")
    print(
        f"changed_ratio_gt_{args.epsilon:g}="
        f"{np.mean(difference > args.epsilon):.9f}"
    )


if __name__ == "__main__":
    main()
