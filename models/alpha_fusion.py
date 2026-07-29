from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn


@dataclass(frozen=True)
class AlphaFusionConfig:
    confidence_high: float = 0.5
    confidence_low: float = 0.5
    residual_std_factor: float = 1.0
    alpha_scale: float = 1.0
    fixed_alpha: Optional[float] = None
    fixed_sigma_bins: Optional[float] = None
    min_alignment_points: int = 4
    eps: float = 1e-6

    def __post_init__(self) -> None:
        for name, value in (
            ("confidence_high", self.confidence_high),
            ("confidence_low", self.confidence_low),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1], got {value}")
        if self.residual_std_factor < 0:
            raise ValueError("residual_std_factor must be non-negative")
        if self.alpha_scale < 0:
            raise ValueError("alpha_scale must be non-negative")
        if self.fixed_sigma_bins is not None and self.fixed_sigma_bins <= 0:
            raise ValueError("fixed_sigma_bins must be positive")
        if self.min_alignment_points < 4:
            raise ValueError("min_alignment_points must be at least 4")


@dataclass
class AlphaFusionOutput:
    logits: Tensor
    probabilities: Tensor
    alpha: Tensor
    sigma_bins: Tensor
    prior_mask: Tensor
    outlier_mask: Tensor
    confidence: Tensor
    mvs_depth: Tensor
    aligned_mono_depth: Tensor
    residual: Tensor


class AlphaNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 8,
        hidden_channels: int = 32,
        init_alpha_bias: float = -2.0,
    ) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.head_alpha = nn.Conv2d(hidden_channels, 1, 1)
        self.head_sigma = nn.Conv2d(hidden_channels, 1, 1)
        nn.init.constant_(self.head_alpha.bias, init_alpha_bias)
        nn.init.constant_(self.head_sigma.bias, 0.0)

    def forward(self, features: Tensor) -> tuple[Tensor, Tensor]:
        hidden = self.body(features)
        alpha = torch.sigmoid(self.head_alpha(hidden))
        sigma_bins = F.softplus(self.head_sigma(hidden)) + 1e-6
        return alpha, sigma_bins


def _as_bhw(depth: Tensor, name: str) -> Tensor:
    if depth.ndim == 3:
        return depth
    if depth.ndim == 4 and depth.shape[1] == 1:
        return depth[:, 0]
    raise ValueError(f"{name} must have shape [B,H,W] or [B,1,H,W], got {tuple(depth.shape)}")


def _expand_depth_values(
    depth_values: Tensor,
    batch: int,
    hypotheses: int,
    height: int,
    width: int,
) -> Tensor:
    if depth_values.ndim == 1:
        depth_values = depth_values.view(1, hypotheses, 1, 1)
    elif depth_values.ndim == 2:
        depth_values = depth_values[:, :, None, None]
    elif depth_values.ndim == 3 and depth_values.shape[0] == hypotheses:
        depth_values = depth_values.unsqueeze(0)
    elif depth_values.ndim != 4:
        raise ValueError(
            "depth_values must have shape [D], [B,D], [D,H,W], or [B,D,H,W]"
        )

    expected = (batch, hypotheses, height, width)
    for actual, target in zip(depth_values.shape, expected):
        if actual not in (1, target):
            raise ValueError(
                f"depth_values shape {tuple(depth_values.shape)} cannot broadcast to {expected}"
            )
    return depth_values.expand(expected)


def depth_to_points(depth: Tensor, intrinsics: Tensor) -> Tensor:
    depth = _as_bhw(depth, "depth")
    batch, height, width = depth.shape
    if intrinsics.ndim == 2:
        intrinsics = intrinsics.unsqueeze(0)
    if intrinsics.shape != (batch, 3, 3):
        raise ValueError(
            f"intrinsics must have shape [B,3,3], got {tuple(intrinsics.shape)}"
        )

    y, x = torch.meshgrid(
        torch.arange(height, device=depth.device, dtype=depth.dtype),
        torch.arange(width, device=depth.device, dtype=depth.dtype),
        indexing="ij",
    )
    x = x.unsqueeze(0)
    y = y.unsqueeze(0)
    fx = intrinsics[:, 0, 0, None, None]
    fy = intrinsics[:, 1, 1, None, None]
    cx = intrinsics[:, 0, 2, None, None]
    cy = intrinsics[:, 1, 2, None, None]

    z = depth
    point_x = (x - cx) * z / fx
    point_y = (y - cy) * z / fy
    return torch.stack((point_x, point_y, z), dim=-1)


def align_depth_scale_shift(
    mono_depth: Tensor,
    mvs_depth: Tensor,
    min_points: int = 4,
) -> Tensor:
    mono_depth = _as_bhw(mono_depth, "mono_depth")
    mvs_depth = _as_bhw(mvs_depth, "mvs_depth")
    if mono_depth.shape != mvs_depth.shape:
        raise ValueError("mono_depth and mvs_depth must have the same shape")

    aligned = mono_depth.float().clone()
    mono_flat = mono_depth.float().flatten(1)
    mvs_flat = mvs_depth.float().flatten(1)
    valid = (
        torch.isfinite(mono_flat)
        & torch.isfinite(mvs_flat)
        & (mono_flat > 0)
        & (mvs_flat > 0)
    )

    for batch_index in range(mono_depth.shape[0]):
        x = mono_flat[batch_index, valid[batch_index]]
        y = mvs_flat[batch_index, valid[batch_index]]
        if x.numel() < min_points:
            continue
        design = torch.stack((x, torch.ones_like(x)), dim=1)
        try:
            solution = torch.linalg.lstsq(design, y[:, None]).solution.flatten()
        except RuntimeError:
            continue
        if solution.numel() < 2 or not bool(torch.isfinite(solution).all()):
            continue
        aligned[batch_index] = (
            solution[0] * mono_depth[batch_index].float() + solution[1]
        )
    return aligned


def align_points_affine(
    mono_points: Tensor,
    mvs_points: Tensor,
    mask: Tensor,
    min_points: int = 4,
) -> Tensor:
    if mono_points.shape != mvs_points.shape or mono_points.ndim != 4:
        raise ValueError("point maps must both have shape [B,H,W,3]")
    if mono_points.shape[-1] != 3 or mask.shape != mono_points.shape[:-1]:
        raise ValueError("mask must have shape [B,H,W]")

    aligned_batches = []
    for batch_index in range(mono_points.shape[0]):
        mono = mono_points[batch_index].float()
        mvs = mvs_points[batch_index].float()
        valid = mask[batch_index]
        source = mono[valid]
        target = mvs[valid]
        if source.shape[0] < min_points:
            aligned_batches.append(mono)
            continue

        ones = torch.ones(
            source.shape[0], 1, device=source.device, dtype=source.dtype
        )
        source_h = torch.cat((source, ones), dim=1)
        try:
            transform = torch.linalg.lstsq(source_h, target).solution
        except RuntimeError:
            aligned_batches.append(mono)
            continue
        if transform.shape != (4, 3) or not bool(torch.isfinite(transform).all()):
            aligned_batches.append(mono)
            continue

        flat = mono.reshape(-1, 3)
        flat_h = torch.cat(
            (
                flat,
                torch.ones(flat.shape[0], 1, device=flat.device, dtype=flat.dtype),
            ),
            dim=1,
        )
        aligned = (flat_h @ transform).reshape_as(mono)
        aligned_batches.append(
            torch.nan_to_num(aligned, nan=0.0, posinf=0.0, neginf=0.0)
        )
    return torch.stack(aligned_batches)


class AlphaFusion(nn.Module):
    def __init__(
        self,
        config: Optional[AlphaFusionConfig] = None,
        alpha_net: Optional[AlphaNet] = None,
    ) -> None:
        super().__init__()
        self.config = config or AlphaFusionConfig()
        self.alpha_net = alpha_net or AlphaNet()

    def forward(
        self,
        mvs_logits: Tensor,
        mono_depth: Tensor,
        depth_values: Tensor,
        intrinsics: Tensor,
    ) -> AlphaFusionOutput:
        if mvs_logits.ndim != 4:
            raise ValueError("mvs_logits must have shape [B,D,H,W]")
        batch, hypotheses, height, width = mvs_logits.shape
        mono_depth = _as_bhw(mono_depth, "mono_depth")
        if mono_depth.shape != (batch, height, width):
            raise ValueError(
                f"mono_depth shape {tuple(mono_depth.shape)} does not match "
                f"{(batch, height, width)}"
            )
        if intrinsics.ndim == 2:
            intrinsics = intrinsics.unsqueeze(0)
        if intrinsics.shape != (batch, 3, 3):
            raise ValueError("intrinsics must have shape [B,3,3]")

        stats_logits = mvs_logits.float()
        depth_values_full = _expand_depth_values(
            depth_values.float(), batch, hypotheses, height, width
        ).to(mvs_logits.device)
        intrinsics_float = intrinsics.to(device=mvs_logits.device, dtype=torch.float32)
        mono_depth_float = mono_depth.to(device=mvs_logits.device, dtype=torch.float32)

        with torch.no_grad():
            probabilities = torch.softmax(stats_logits, dim=1)
            confidence, mvs_index = probabilities.max(dim=1)
            top2 = torch.topk(probabilities, k=2, dim=1).values
            mvs_depth = torch.gather(
                depth_values_full, 1, mvs_index[:, None]
            ).squeeze(1)

            mono_scale_shift = align_depth_scale_shift(
                mono_depth_float,
                mvs_depth,
                min_points=self.config.min_alignment_points,
            )
            mono_valid = torch.isfinite(mono_scale_shift) & (mono_scale_shift > 0)
            high_mask = (confidence > self.config.confidence_high) & mono_valid

            mvs_points = depth_to_points(mvs_depth, intrinsics_float)
            mono_points = depth_to_points(mono_scale_shift, intrinsics_float)
            mono_points = torch.nan_to_num(
                mono_points, nan=0.0, posinf=0.0, neginf=0.0
            )
            aligned_points = align_points_affine(
                mono_points,
                mvs_points,
                high_mask,
                min_points=self.config.min_alignment_points,
            )
            aligned_mono_depth = aligned_points[..., 2]

            residual = mvs_depth - aligned_mono_depth
            residual_values = residual[high_mask]
            if (
                residual_values.numel() >= 2
                and bool(torch.isfinite(residual_values).all())
            ):
                residual_mean = residual_values.mean()
                residual_std = residual_values.std() + self.config.eps
                outlier_mask = (
                    high_mask
                    & mono_valid
                    & (
                        (residual - residual_mean).abs()
                        > self.config.residual_std_factor * residual_std
                    )
                )
            else:
                outlier_mask = torch.zeros_like(high_mask)

            low_mask = confidence < self.config.confidence_low
            prior_mask = (low_mask | outlier_mask) & mono_valid
            entropy = -(
                probabilities * (probabilities + self.config.eps).log()
            ).sum(dim=1)
            margin = top2[:, 0] - top2[:, 1]
            features = torch.stack(
                (
                    confidence,
                    entropy,
                    margin,
                    torch.log(mvs_depth.clamp_min(self.config.eps)),
                    torch.log(aligned_mono_depth.clamp_min(self.config.eps)),
                    residual,
                    residual.abs(),
                    prior_mask.float(),
                ),
                dim=1,
            )
            features = torch.nan_to_num(
                features, nan=0.0, posinf=0.0, neginf=0.0
            )

            depth_difference = (
                depth_values_full - aligned_mono_depth[:, None]
            ).abs()
            mono_index = depth_difference.argmin(dim=1, keepdim=True)
            logit_scale = stats_logits.abs().amax(dim=1, keepdim=True)

        parameter_dtype = next(self.alpha_net.parameters()).dtype
        alpha, sigma_bins = self.alpha_net(features.to(parameter_dtype))
        if self.config.fixed_alpha is not None:
            alpha = torch.full_like(alpha, self.config.fixed_alpha)
        alpha = alpha * self.config.alpha_scale
        if self.config.fixed_sigma_bins is not None:
            sigma_bins = torch.full_like(
                sigma_bins, self.config.fixed_sigma_bins
            )
        sigma_bins = sigma_bins + self.config.eps

        hypothesis_index = torch.arange(
            hypotheses,
            device=mvs_logits.device,
            dtype=alpha.dtype,
        ).view(1, hypotheses, 1, 1)
        gaussian = torch.exp(
            -(
                (hypothesis_index - mono_index.to(alpha.dtype)) ** 2
                / (2.0 * sigma_bins**2)
            )
        )
        gaussian = gaussian * prior_mask[:, None].to(alpha.dtype)
        correction = alpha * logit_scale.to(alpha.dtype) * gaussian
        fused_logits = mvs_logits + correction.to(mvs_logits.dtype)
        fused_probabilities = torch.softmax(fused_logits.float(), dim=1)

        return AlphaFusionOutput(
            logits=fused_logits,
            probabilities=fused_probabilities,
            alpha=alpha,
            sigma_bins=sigma_bins,
            prior_mask=prior_mask,
            outlier_mask=outlier_mask,
            confidence=confidence,
            mvs_depth=mvs_depth,
            aligned_mono_depth=aligned_mono_depth,
            residual=residual,
        )


def apply_alpha_fusion(
    mvs_logits: Tensor,
    mono_depth: Tensor,
    depth_values: Tensor,
    intrinsics: Tensor,
    alpha_net: AlphaNet,
    config: AlphaFusionConfig,
) -> AlphaFusionOutput:
    """Apply fusion while keeping AlphaNet registered directly on StageNet."""
    return AlphaFusion(config=config, alpha_net=alpha_net)(
        mvs_logits,
        mono_depth,
        depth_values,
        intrinsics,
    )
