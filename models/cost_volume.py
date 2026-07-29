import os

import torch
import torch.nn.functional as F
from torch import nn

from models.alpha_fusion import (
    AlphaFusionConfig,
    AlphaNet,
    apply_alpha_fusion,
)
from models.module import (
    ConvBnReLU,
    CostRegNet,
    CostRegNet3D,
    PureTransformerCostReg,
    conf_regression,
    depth_regression,
)
from models.warping import homo_warping_3D_with_mask


class _IdentityContext:
    def __init__(self, enabled=True):
        self.enabled = enabled

    def __enter__(self):
        return None

    def __exit__(self, *args):
        return None


autocast = (
    torch.cuda.amp.autocast
    if torch.__version__ >= "1.6.0"
    else _IdentityContext
)


def _optional_env_float(name, default=None):
    value = os.environ.get(name, "").strip()
    return default if value == "" else float(value)


class StageNet(nn.Module):
    """One MVSFormer++ cost-volume stage with MR-MVS AlphaNet correction."""

    def __init__(self, args, ndepth, stage_idx):
        super().__init__()
        self.args = args
        self.ndepth = ndepth
        self.stage_idx = stage_idx
        self.fusion_type = args.get("fusion_type", "cnn")
        self.cost_reg_type = args.get(
            "cost_reg_type", ["Normal", "Normal", "Normal", "Normal"]
        )[stage_idx]
        self.depth_type = args["depth_type"]
        if isinstance(self.depth_type, list):
            self.depth_type = self.depth_type[stage_idx]

        alpha_args = dict(args.get("alpha_fusion", {}))
        enabled_stages = tuple(alpha_args.get("enabled_stages", [1, 2, 3]))
        self.alpha_enabled = bool(alpha_args.get("enabled", True)) and (
            stage_idx + 1 in enabled_stages
        )
        self.alpha_net = AlphaNet(
            in_channels=8,
            hidden_channels=int(alpha_args.get("hidden_channels", 32)),
            init_alpha_bias=float(alpha_args.get("init_alpha_bias", -2.0)),
        )
        self.alpha_config = AlphaFusionConfig(
            confidence_high=_optional_env_float(
                "MVSFORMER_ALPHA_CONF_HIGH_THRESHOLD",
                float(alpha_args.get("confidence_high", 0.5)),
            ),
            confidence_low=_optional_env_float(
                "MVSFORMER_ALPHA_CONF_LOW_THRESHOLD",
                float(alpha_args.get("confidence_low", 0.5)),
            ),
            residual_std_factor=float(
                alpha_args.get("residual_std_factor", 1.0)
            ),
            alpha_scale=_optional_env_float(
                "MVSFORMER_ALPHA_SCALE",
                float(alpha_args.get("alpha_scale", 1.0)),
            ),
            fixed_alpha=_optional_env_float(
                "MVSFORMER_FIXED_ALPHA_GATE",
                alpha_args.get("fixed_alpha"),
            ),
            fixed_sigma_bins=_optional_env_float(
                "MVSFORMER_FIXED_SIGMA",
                alpha_args.get("fixed_sigma_bins"),
            ),
            min_alignment_points=int(
                alpha_args.get("min_alignment_points", 4)
            ),
        )

        in_channels = args["base_ch"]
        if isinstance(in_channels, list):
            in_channels = in_channels[stage_idx]

        if self.fusion_type != "cnn":
            raise NotImplementedError(
                f"Unsupported feature fusion type: {self.fusion_type}"
            )
        self.vis = nn.Sequential(
            ConvBnReLU(1, 16),
            ConvBnReLU(16, 16),
            ConvBnReLU(16, 8),
            nn.Conv2d(8, 1, 1),
            nn.Sigmoid(),
        )

        if self.cost_reg_type == "PureTransformerCostReg":
            args["transformer_config"][stage_idx]["base_channel"] = in_channels
            self.cost_reg = PureTransformerCostReg(
                in_channels,
                **args["transformer_config"][stage_idx],
            )
        else:
            model_threshold = args.get("model_th", 8)
            if ndepth <= model_threshold:
                self.cost_reg = CostRegNet3D(in_channels, in_channels)
            else:
                self.cost_reg = CostRegNet(in_channels, in_channels)

    def forward(
        self,
        features,
        proj_matrices,
        depth_values,
        tmp,
        position3d=None,
        mono_depth=None,
        intrinsics=None,
    ):
        ref_feat = features[:, 0]
        src_feats = torch.unbind(features[:, 1:], dim=1)
        projections = torch.unbind(proj_matrices, dim=1)
        if len(src_feats) != len(projections) - 1:
            raise ValueError("Feature and camera view counts do not match")

        ref_proj, src_projs = projections[0], projections[1:]
        volume_sum = 0.0
        visibility_sum = 0.0
        visibility_weights = []

        with autocast(enabled=False):
            for src_feat, src_proj in zip(src_feats, src_projs):
                src_feat = src_feat.float()
                src_projection = src_proj[:, 0].clone()
                src_projection[:, :3, :4] = torch.matmul(
                    src_proj[:, 1, :3, :3],
                    src_proj[:, 0, :3, :4],
                )
                ref_projection = ref_proj[:, 0].clone()
                ref_projection[:, :3, :4] = torch.matmul(
                    ref_proj[:, 1, :3, :3],
                    ref_proj[:, 0, :3, :4],
                )
                warped_volume, _ = homo_warping_3D_with_mask(
                    src_feat,
                    src_projection,
                    ref_projection,
                    depth_values,
                )

                batch, channels, hypotheses, height, width = (
                    warped_volume.shape
                )
                groups = self.args["base_ch"]
                if isinstance(groups, list):
                    groups = groups[self.stage_idx]

                if groups < channels:
                    warped_volume = warped_volume.view(
                        batch,
                        groups,
                        channels // groups,
                        hypotheses,
                        height,
                        width,
                    )
                    ref_volume = (
                        ref_feat.view(
                            batch,
                            groups,
                            channels // groups,
                            1,
                            height,
                            width,
                        )
                        .repeat(1, 1, 1, hypotheses, 1, 1)
                        .float()
                    )
                    inner_product = (ref_volume * warped_volume).mean(dim=2)
                elif groups == channels:
                    ref_volume = ref_feat.view(
                        batch, groups, 1, height, width
                    ).float()
                    inner_product = ref_volume * warped_volume
                else:
                    raise ValueError("base_ch must not exceed feature channels")

                similarity = inner_product.sum(dim=1)
                similarity_probability = F.softmax(
                    similarity.detach(), dim=1
                )
                entropy = -(
                    similarity_probability
                    * torch.log(similarity_probability + 1e-7)
                ).sum(dim=1, keepdim=True)
                visibility = self.vis(entropy)
                visibility_weights.append(visibility)
                volume_sum = (
                    volume_sum + inner_product * visibility.unsqueeze(1)
                )
                visibility_sum = visibility_sum + visibility

            volume_mean = volume_sum / (
                visibility_sum.unsqueeze(1) + 1e-6
            )

        raw_logits = self.cost_reg(volume_mean, position3d).squeeze(1)
        alpha_output = None
        logits = raw_logits
        if self.alpha_enabled and mono_depth is not None:
            if intrinsics is None:
                raise ValueError(
                    "Stage intrinsics are required for AlphaNet fusion"
                )
            alpha_output = apply_alpha_fusion(
                raw_logits,
                mono_depth,
                depth_values,
                intrinsics,
                self.alpha_net,
                self.alpha_config,
            )
            logits = alpha_output.logits

        probability = F.softmax(logits, dim=1)
        if self.depth_type == "ce":
            if self.training:
                depth_index = probability.argmax(dim=1)
                depth = torch.gather(
                    depth_values, 1, depth_index.unsqueeze(1)
                ).squeeze(1)
                confidence = F.softmax(raw_logits, dim=1).max(dim=1)[0]
            else:
                depth = depth_regression(
                    F.softmax(logits * tmp, dim=1),
                    depth_values=depth_values,
                )
                confidence = probability.max(dim=1)[0]
        else:
            depth = depth_regression(probability, depth_values=depth_values)
            if self.ndepth >= 32:
                confidence = conf_regression(probability, n=4)
            elif self.ndepth == 16:
                confidence = conf_regression(probability, n=3)
            elif self.ndepth == 8:
                confidence = conf_regression(probability, n=2)
            else:
                confidence = probability.max(dim=1)[0]

        outputs = {
            "depth": depth,
            "prob_volume": probability,
            "photometric_confidence": confidence.detach(),
            "depth_values": depth_values,
            "prob_volume_pre": logits,
            "vis_list": visibility_weights,
        }
        if alpha_output is not None:
            outputs.update(
                {
                    "alpha": alpha_output.alpha,
                    "sigma_bins": alpha_output.sigma_bins,
                    "prior_mask": alpha_output.prior_mask,
                    "aligned_mono_depth": alpha_output.aligned_mono_depth,
                }
            )
        return outputs
