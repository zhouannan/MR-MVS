import os

import torch
import torch.nn.functional as F
from torch import nn

from models.cost_volume import StageNet
from models.dino.dinov2 import vit_base
from models.FMT import FMT_with_pathway
from models.module import (
    CrossVITDecoder,
    FPNDecoder,
    FPNEncoder,
    init_inverse_range,
    init_range,
    schedule_inverse_range,
    schedule_range,
)
from models.position_encoding import get_position_3d
from models.stage_refine import refine_stage_depth_with_planes


ALIGN_CORNERS = False


class DINOv2MVSNet(nn.Module):
    """MVSFormer++ backbone with the two MR-MVS components only."""

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.ndepths = args["ndepths"]
        self.depth_intervals_ratio = args["depth_interals_ratio"]
        self.inverse_depth = args.get("inverse_depth", False)
        self.use_pe3d = args.get("use_pe3d", False)
        self.cost_reg_type = args.get(
            "cost_reg_type", ["Normal", "Normal", "Normal", "Normal"]
        )

        self.stage_refine_reproj_threshold = float(
            args.get("stage_refine_reproj_threshold", 0.01)
        )
        self.stage_refine_iters = max(
            0, int(args.get("stage_refine_iters", 10))
        )
        self.stage_refine_use_noise = bool(
            int(args.get("stage_refine_use_noise", 1))
        )
        self.stage_refine_max_srcs = int(
            args.get("stage_refine_max_srcs", 0)
        )

        self.encoder = FPNEncoder(feat_chs=args["feat_chs"])
        self.decoder = FPNDecoder(feat_chs=args["feat_chs"])
        self.freeze_vit = bool(args["freeze_vit"])
        self.vit = vit_base(
            img_size=518,
            patch_size=14,
            init_values=1.0,
            block_chunks=0,
            ffn_layer="mlp",
            **args.get("dino_cfg", {}),
        )
        self.decoder_vit = CrossVITDecoder(args)
        self.FMT_module = FMT_with_pathway(**args.get("FMT_config"))

        vit_path = os.path.expanduser(args["vit_path"])
        if os.path.isfile(vit_path):
            checkpoint = torch.load(vit_path, map_location="cpu")
            from utils import torch_init_model

            torch_init_model(self.vit, checkpoint, key="model")
        else:
            print(
                f"Warning: DINOv2 weights not found at {vit_path}. "
                "Model construction continues for inspection only."
            )

        self.fusions = nn.ModuleList(
            [
                StageNet(args, self.ndepths[index], index)
                for index in range(len(self.ndepths))
            ]
        )
        if len(self.fusions) >= 4:
            for parameter in self.fusions[3].alpha_net.parameters():
                parameter.requires_grad = False

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_vit:
            self.vit.eval()
        return self

    @staticmethod
    def _resize_mono_depth(mono_depth, stage_index, target_hw):
        if mono_depth is None:
            return None
        if isinstance(mono_depth, dict):
            mono_depth = mono_depth.get(f"stage{stage_index + 1}")
            if mono_depth is None:
                return None
        if mono_depth.ndim == 3:
            mono_depth = mono_depth.unsqueeze(1)
        if mono_depth.ndim != 4:
            raise ValueError(
                "mono_depth must be [B,H,W], [B,1,H,W], or a stage dict"
            )
        if mono_depth.shape[-2:] != tuple(target_hw):
            mono_depth = F.interpolate(
                mono_depth.float(),
                size=target_hw,
                mode="bilinear",
                align_corners=False,
            )
        return mono_depth

    def vit_forward(self, vit_images, batch, views, vit_h, vit_w):
        if self.freeze_vit:
            with torch.no_grad():
                interval_features = self.vit.forward_interval_features(
                    vit_images
                )
        else:
            interval_features = self.vit.forward_interval_features(vit_images)

        interval_features = [
            feature.reshape(batch, views, -1, self.vit.embed_dim)
            for feature in interval_features
        ]
        vit_shape = [
            batch,
            views,
            vit_h // self.vit.patch_size,
            vit_w // self.vit.patch_size,
            self.vit.embed_dim,
        ]
        return self.decoder_vit(
            interval_features,
            Fmats=None,
            vit_shape=vit_shape,
        )

    def _extract_features(self, images):
        batch, views, _, height, width = images.shape
        vit_h = int(
            height * self.args["rescale"] // self.vit.patch_size
            * self.vit.patch_size
        )
        vit_w = int(
            width * self.args["rescale"] // self.vit.patch_size
            * self.vit.patch_size
        )
        flat_images = images.reshape(batch * views, 3, height, width)
        vit_images = F.interpolate(
            flat_images,
            (vit_h, vit_w),
            mode="bicubic",
            align_corners=ALIGN_CORNERS,
        )
        vit_feature = self.vit_forward(
            vit_images, batch, views, vit_h, vit_w
        )

        if self.training:
            conv01, conv11, conv21, conv31 = self.encoder(flat_images)
            if vit_feature.shape[-2:] != conv31.shape[-2:]:
                vit_feature = F.interpolate(
                    vit_feature,
                    size=conv31.shape[-2:],
                    mode="bilinear",
                    align_corners=ALIGN_CORNERS,
                )
            feat1, feat2, feat3, feat4 = self.decoder(
                conv01,
                conv11,
                conv21,
                conv31 + vit_feature,
            )
            return {
                "stage1": feat1.reshape(
                    batch, views, *feat1.shape[1:]
                ),
                "stage2": feat2.reshape(
                    batch, views, *feat2.shape[1:]
                ),
                "stage3": feat3.reshape(
                    batch, views, *feat3.shape[1:]
                ),
                "stage4": feat4.reshape(
                    batch, views, *feat4.shape[1:]
                ),
            }

        target_hw = (height // 8, width // 8)
        if vit_feature.shape[-2:] != target_hw:
            vit_feature = F.interpolate(
                vit_feature,
                size=target_hw,
                mode="bilinear",
                align_corners=ALIGN_CORNERS,
            )
        vit_feature = vit_feature.reshape(
            batch, views, *vit_feature.shape[1:]
        )
        stage_features = [[], [], [], []]
        for view_index in range(views):
            conv01, conv11, conv21, conv31 = self.encoder(
                images[:, view_index]
            )
            decoded = self.decoder(
                conv01,
                conv11,
                conv21,
                conv31 + vit_feature[:, view_index],
            )
            for stage_index, feature in enumerate(decoded):
                stage_features[stage_index].append(feature)
        return {
            f"stage{index + 1}": torch.stack(features, dim=1)
            for index, features in enumerate(stage_features)
        }

    def forward(
        self,
        imgs,
        proj_matrices,
        depth_values,
        mono_depth=None,
        tmp=(5.0, 5.0, 5.0, 1.0),
    ):
        batch, _, _, image_h, image_w = imgs.shape
        depth_interval = depth_values[:, 1] - depth_values[:, 0]
        features = self.FMT_module(self._extract_features(imgs))

        outputs = {}
        previous = {}
        height_min = height_max = None
        width_min = width_max = None
        confidence_sum = torch.zeros(
            batch,
            image_h,
            image_w,
            dtype=torch.float32,
            device=imgs.device,
        )

        for stage_index in range(len(self.ndepths)):
            stage_name = f"stage{stage_index + 1}"
            stage_cameras = proj_matrices[stage_name]
            stage_features = features[stage_name]
            _, _, _, height, width = stage_features.shape
            intrinsics = stage_cameras[:, 0, 1, :3, :3]

            if stage_index == 0:
                if self.inverse_depth:
                    depth_samples = init_inverse_range(
                        depth_values,
                        self.ndepths[stage_index],
                        imgs.device,
                        imgs.dtype,
                        height,
                        width,
                    )
                else:
                    depth_samples = init_range(
                        depth_values,
                        self.ndepths[stage_index],
                        imgs.device,
                        imgs.dtype,
                        height,
                        width,
                    )
            elif self.inverse_depth:
                depth_samples = schedule_inverse_range(
                    previous["depth"].detach(),
                    previous["depth_values"],
                    self.ndepths[stage_index],
                    self.depth_intervals_ratio[stage_index],
                    height,
                    width,
                )
            else:
                depth_samples = schedule_range(
                    previous["depth"].detach(),
                    self.ndepths[stage_index],
                    self.depth_intervals_ratio[stage_index]
                    * depth_interval,
                    height,
                    width,
                )

            use_position = (
                self.cost_reg_type[stage_index] != "Normal"
                and self.use_pe3d
            )
            if use_position:
                (
                    position3d,
                    height_min,
                    height_max,
                    width_min,
                    width_max,
                ) = get_position_3d(
                    batch,
                    height,
                    width,
                    intrinsics,
                    depth_samples,
                    depth_min=depth_values.min(),
                    depth_max=depth_values.max(),
                    height_min=height_min,
                    height_max=height_max,
                    width_min=width_min,
                    width_max=width_max,
                    normalize=True,
                )
            else:
                position3d = None

            stage_mono = self._resize_mono_depth(
                mono_depth,
                stage_index,
                (height, width),
            )
            previous = self.fusions[stage_index](
                stage_features,
                stage_cameras,
                depth_samples,
                tmp=tmp[stage_index],
                position3d=position3d,
                mono_depth=stage_mono if stage_index < 3 else None,
                intrinsics=intrinsics,
            )
            if stage_mono is not None:
                previous["mono_depth"] = stage_mono.squeeze(1)
            outputs[stage_name] = previous
            outputs.update(previous)

            stage_confidence = previous["photometric_confidence"]
            if stage_confidence.shape[-2:] != (image_h, image_w):
                stage_confidence = F.interpolate(
                    stage_confidence.unsqueeze(1),
                    size=(image_h, image_w),
                    mode="nearest",
                ).squeeze(1)
            confidence_sum += stage_confidence

        outputs["refined_depth"] = previous["depth"]
        outputs["photometric_confidence"] = confidence_sum / len(
            self.ndepths
        )
        return outputs

    def refine_stage_depth(
        self,
        imgs,
        proj_matrices,
        depths,
        pre_output,
        depth_values,
        stage_idx,
    ):
        return refine_stage_depth_with_planes(
            imgs=imgs,
            proj_matrices=proj_matrices,
            depths=depths,
            pre_output=pre_output,
            depth_values=depth_values,
            stage_idx=stage_idx,
            reproj_threshold=self.stage_refine_reproj_threshold,
            refine_iters=self.stage_refine_iters,
            use_noise=self.stage_refine_use_noise,
            max_srcs=self.stage_refine_max_srcs,
        )
