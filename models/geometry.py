import torch
import torch.nn.functional as F


def compute_normals_from_depth(depth, intrinsics):
    batch, _, height, width = depth.shape
    device = depth.device
    dtype = depth.dtype

    y, x = torch.meshgrid(
        torch.arange(height, device=device, dtype=dtype),
        torch.arange(width, device=device, dtype=dtype),
        indexing="ij",
    )
    x = x.view(1, 1, height, width)
    y = y.view(1, 1, height, width)
    fx = intrinsics[:, 0, 0].view(batch, 1, 1, 1)
    fy = intrinsics[:, 1, 1].view(batch, 1, 1, 1)
    cx = intrinsics[:, 0, 2].view(batch, 1, 1, 1)
    cy = intrinsics[:, 1, 2].view(batch, 1, 1, 1)

    z = depth
    points = torch.cat(((x - cx) * z / fx, (y - cy) * z / fy, z), dim=1)
    dx = F.pad(points[:, :, :, 1:] - points[:, :, :, :-1], (0, 1))
    dy = F.pad(points[:, :, 1:, :] - points[:, :, :-1, :], (0, 0, 0, 1))
    return F.normalize(torch.cross(dx, dy, dim=1), dim=1)


def compute_rays(height, width, intrinsics):
    batch = intrinsics.shape[0]
    device = intrinsics.device
    dtype = intrinsics.dtype
    y, x = torch.meshgrid(
        torch.arange(height, device=device, dtype=dtype),
        torch.arange(width, device=device, dtype=dtype),
        indexing="ij",
    )
    x = x.view(1, 1, height, width)
    y = y.view(1, 1, height, width)
    fx = intrinsics[:, 0, 0].view(batch, 1, 1, 1)
    fy = intrinsics[:, 1, 1].view(batch, 1, 1, 1)
    cx = intrinsics[:, 0, 2].view(batch, 1, 1, 1)
    cy = intrinsics[:, 1, 2].view(batch, 1, 1, 1)
    return torch.cat(((x - cx) / fx, (y - cy) / fy, torch.ones_like(x)), dim=1)


def plane_from_depth_normal(depth, normal, intrinsics):
    points = compute_rays(depth.shape[-2], depth.shape[-1], intrinsics) * depth
    plane_d = -(normal * points).sum(dim=1, keepdim=True)
    return normal, plane_d


def propagate_planes(plane_n, plane_d):
    n_up = F.pad(plane_n[:, :, :-1, :], (0, 0, 1, 0))
    d_up = F.pad(plane_d[:, :, :-1, :], (0, 0, 1, 0))
    n_down = F.pad(plane_n[:, :, 1:, :], (0, 0, 0, 1))
    d_down = F.pad(plane_d[:, :, 1:, :], (0, 0, 0, 1))
    n_left = F.pad(plane_n[:, :, :, :-1], (1, 0, 0, 0))
    d_left = F.pad(plane_d[:, :, :, :-1], (1, 0, 0, 0))
    n_right = F.pad(plane_n[:, :, :, 1:], (0, 1, 0, 0))
    d_right = F.pad(plane_d[:, :, :, 1:], (0, 1, 0, 0))
    return (
        torch.stack((plane_n, n_up, n_down, n_left, n_right), dim=1),
        torch.stack((plane_d, d_up, d_down, d_left, d_right), dim=1),
    )


def plane_to_depth(plane_n, plane_d, intrinsics):
    rays = compute_rays(plane_n.shape[-2], plane_n.shape[-1], intrinsics)
    denominator = (plane_n * rays).sum(dim=1)
    return (-plane_d[:, 0] / (denominator + 1e-9)).unsqueeze(1)
