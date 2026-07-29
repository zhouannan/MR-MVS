import torch
import torch.nn.functional as F

from models.geometry import (
    compute_normals_from_depth,
    compute_rays,
    plane_from_depth_normal,
    plane_to_depth,
    propagate_planes,
)


def _as_bchw(tensor, batch_size):
    if tensor.dim() == 2:
        return tensor.unsqueeze(0).unsqueeze(0)
    if tensor.dim() == 3:
        if tensor.shape[0] == batch_size:
            return tensor.unsqueeze(1)
        return tensor.unsqueeze(0)
    if tensor.dim() == 4:
        return tensor
    raise ValueError(f"Expected 2D/3D/4D tensor, got shape {tuple(tensor.shape)}")


def plane_candidates_to_depth(all_n, all_d, ref_K):
    rays = compute_rays(all_n.shape[-2], all_n.shape[-1], ref_K).unsqueeze(1)
    denom = (all_n * rays).sum(dim=2)
    depth = -all_d.squeeze(2) / (denom + 1e-9)
    return depth.unsqueeze(2)


def build_reproject_context(height, width, ref_K, ref_E, src_Ks, src_projs):
    B = ref_K.shape[0]
    device = ref_K.device
    dtype = ref_K.dtype
    y, x = torch.meshgrid(
        torch.arange(height, device=device, dtype=dtype),
        torch.arange(width, device=device, dtype=dtype),
        indexing="ij",
    )
    xy_homo = torch.stack([x, y, torch.ones_like(x)], dim=0).view(1, 3, -1).expand(B, -1, -1)
    ones = torch.ones(B, 1, height * width, device=device, dtype=dtype)
    ref_K_inv = torch.inverse(ref_K)
    src_contexts = []
    for src_K, src_proj in zip(src_Ks, src_projs):
        src_E = src_proj[:, 0, :4, :4]
        src_contexts.append(
            {
                "K": src_K,
                "K_inv": torch.inverse(src_K),
                "E": src_E,
                "E_inv": torch.inverse(src_E),
                "P": src_K @ src_E[:, :3, :],
            }
        )
    return {
        "height": height,
        "width": width,
        "xy_homo": xy_homo,
        "xy_ref": xy_homo[:, :2],
        "ref_rays": ref_K_inv @ xy_homo,
        "ones": ones,
        "ref_K": ref_K,
        "ref_K_inv": ref_K_inv,
        "ref_E": ref_E,
        "ref_E_inv": torch.inverse(ref_E),
        "ref_P": ref_K @ ref_E[:, :3, :],
        "src": src_contexts,
    }


def compute_reproject_error_cached(depth_ref, depth_src, context, src_context):
    B, _, H, W = depth_ref.shape
    ones = context["ones"]

    depth_flat = depth_ref.view(B, 1, -1)
    pts_cam_ref = context["ref_rays"] * depth_flat
    pts_cam_ref_homo = torch.cat([pts_cam_ref, ones], dim=1)
    pts_world = context["ref_E_inv"] @ pts_cam_ref_homo

    pts_proj_src = src_context["P"] @ pts_world
    pts_proj_src = pts_proj_src[:, :2] / (pts_proj_src[:, 2:3] + 1e-8)

    u_src = pts_proj_src[:, 0]
    v_src = pts_proj_src[:, 1]
    valid_proj = (u_src >= 0) & (u_src < W) & (v_src >= 0) & (v_src < H)

    u_src_norm = (u_src / (W - 1)) * 2 - 1
    v_src_norm = (v_src / (H - 1)) * 2 - 1
    grid = torch.stack([u_src_norm, v_src_norm], dim=-1).view(B, H, W, 2)
    depth_src_sampled = F.grid_sample(
        depth_src,
        grid,
        mode="bilinear",
        align_corners=True,
    ).view(B, -1)

    pts_proj_src_homo = torch.cat([pts_proj_src, ones], dim=1)
    pts_cam_src_rebuilt = src_context["K_inv"] @ (
        pts_proj_src_homo * depth_src_sampled.unsqueeze(1)
    )
    pts_cam_src_rebuilt_h = torch.cat([pts_cam_src_rebuilt, ones], dim=1)
    pts_world_rebuilt = src_context["E_inv"] @ pts_cam_src_rebuilt_h

    pts_ref_reproj = context["ref_P"] @ pts_world_rebuilt
    pts_ref_reproj = pts_ref_reproj[:, :2] / (pts_ref_reproj[:, 2:3] + 1e-8)
    dist = torch.norm(pts_ref_reproj - context["xy_ref"], dim=1).view(B, 1, H, W)
    dist[~valid_proj.view(B, 1, H, W)] = 1e6
    return dist


def compute_multi_view_weighted_error_cached(
    depth,
    src_depths,
    vis_list,
    reproject_context,
    threshold=0.01,
):
    B = depth.shape[0]
    error_sum = depth.new_zeros(depth.shape)
    weight_sum = depth.new_zeros(depth.shape)

    for src_depth, vis_weight, src_context in zip(src_depths, vis_list, reproject_context["src"]):
        src_depth_bchw = _as_bchw(src_depth, B).to(device=depth.device, dtype=depth.dtype)
        vis_weight_bchw = _as_bchw(vis_weight, B).to(device=depth.device, dtype=depth.dtype)
        err = compute_reproject_error_cached(depth, src_depth_bchw, reproject_context, src_context)
        valid_mask = (err <= threshold).float()
        masked_weight = vis_weight_bchw * valid_mask
        error_sum += err * valid_mask * masked_weight
        weight_sum += masked_weight

    fused_error = error_sum / (weight_sum + 1e-6)
    return fused_error, weight_sum


def refine_stage_depth_with_planes(
    *,
    imgs,
    proj_matrices,
    depths,
    pre_output,
    depth_values,
    stage_idx,
    reproj_threshold,
    refine_iters,
    use_noise,
    max_srcs,
):
    B, H, W = imgs.shape[0], imgs.shape[3], imgs.shape[4]
    proj_matrices_stage = proj_matrices["stage{}".format(stage_idx + 1)]
    ref_K = proj_matrices_stage[:, 0, 1, :3, :3]
    srcs_K = proj_matrices_stage[:, 1:, 1, :3, :3]
    src_count = srcs_K.shape[1]
    if max_srcs > 0:
        src_count = min(src_count, max_srcs)
        proj_matrices_stage = torch.cat(
            [proj_matrices_stage[:, :1], proj_matrices_stage[:, 1:1 + src_count]],
            dim=1,
        )
        srcs_K = srcs_K[:, :src_count]
    srcs_K_list = [srcs_K[:, s] for s in range(src_count)]

    ref_proj, *src_projs = torch.unbind(proj_matrices_stage, 1)
    ref_R = ref_proj[:, 0, :4, :4]
    depth = depths[0].unsqueeze(0).unsqueeze(0)
    src_depths = depths[1:]
    vis_list = pre_output["vis_list"]
    if max_srcs > 0:
        src_depths = src_depths[:src_count]
        vis_list = vis_list[:src_count]
    src_depths = [
        _as_bchw(src_depth, depth.shape[0]).to(device=depth.device, dtype=depth.dtype)
        for src_depth in src_depths
    ]
    vis_list = [
        _as_bchw(vis_weight, depth.shape[0]).to(device=depth.device, dtype=depth.dtype)
        for vis_weight in vis_list
    ]

    normals = compute_normals_from_depth(depth, ref_K)
    plane_n, plane_d = plane_from_depth_normal(depth, normals, ref_K)
    reproject_context = build_reproject_context(
        depth.shape[-2], depth.shape[-1], ref_K, ref_R, srcs_K_list, src_projs
    )

    for _ in range(refine_iters):
        cand_n, cand_d = propagate_planes(plane_n, plane_d)

        if use_noise:
            noise_n = F.normalize(plane_n + 0.01 * torch.randn_like(plane_n), dim=1)
            noise_d = plane_d + 0.01 * torch.randn_like(plane_d)
            all_n = torch.cat([cand_n, noise_n.unsqueeze(1)], dim=1)
            all_d = torch.cat([cand_d, noise_d.unsqueeze(1)], dim=1)
        else:
            all_n = cand_n
            all_d = cand_d
        origin_error, origin_weight = compute_multi_view_weighted_error_cached(
            depth, src_depths, vis_list, reproject_context, threshold=reproj_threshold
        )
        origin_error[origin_weight == 0] = 1e6
        all_depths = plane_candidates_to_depth(all_n, all_d, ref_K)
        B, K, _, H, W = all_depths.shape
        all_errors = torch.empty(B, K, 1, H, W, device=depth.device, dtype=depth.dtype)
        for k in range(K):
            error_k, weight_k = compute_multi_view_weighted_error_cached(
                all_depths[:, k], src_depths, vis_list, reproject_context, threshold=reproj_threshold
            )
            error_k[weight_k == 0] = 1e6
            all_errors[:, k] = error_k
        best_idx = torch.argmin(all_errors, dim=1, keepdim=True)
        best_error = torch.gather(all_errors, 1, best_idx)
        best_depth = torch.gather(all_depths, 1, best_idx)
        replace_mask = (best_error < origin_error).squeeze(1)
        best_depth = best_depth.squeeze(1)
        depth_final = depth.clone()
        depth_final[replace_mask] = best_depth[replace_mask]
        normals_final = compute_normals_from_depth(depth_final, ref_K)
        plane_n, plane_d = plane_from_depth_normal(depth_final, normals_final, ref_K)

    depth_refine = plane_to_depth(plane_n, plane_d, ref_K)
    depth_refine = torch.clamp(depth_refine, min=depth_values.min(), max=depth_values.max())
    return depth_refine.squeeze(1)
