import torch
import torch.nn.functional as F


class identity_with(object):
    def __init__(self, enabled=True):
        self._enabled = enabled

    def __enter__(self):
        pass

    def __exit__(self, *args):
        pass


autocast = torch.cuda.amp.autocast if torch.__version__ >= '1.6.0' else identity_with


def global_relative_depth_ranking_loss(
    pred_depth,
    mono_depth,
    valid_mask,
    sample_n=4096,
    min_pairs=128,
    min_depth=1e-3,
    mono_eps=1e-4,
):
    """
    Pairwise ranking loss:
    if mono says pixel A is closer than pixel B, pred should keep that order.
    """
    if mono_depth is None or pred_depth is None or valid_mask is None:
        return pred_depth.new_tensor(0.0) if pred_depth is not None else torch.tensor(0.0)

    if mono_depth.ndim == 4 and mono_depth.shape[1] == 1:
        mono_depth = mono_depth.squeeze(1)
    if valid_mask.ndim == 4 and valid_mask.shape[1] == 1:
        valid_mask = valid_mask.squeeze(1)

    if mono_depth.shape[-2:] != pred_depth.shape[-2:]:
        mono_depth = F.interpolate(
            mono_depth.unsqueeze(1),
            size=pred_depth.shape[-2:],
            mode='bilinear',
            align_corners=False
        ).squeeze(1)

    valid = (
        valid_mask.bool() &
        torch.isfinite(pred_depth) &
        torch.isfinite(mono_depth) &
        (pred_depth > min_depth) &
        (mono_depth > min_depth)
    )

    pred_vec = pred_depth[valid]
    mono_vec = mono_depth[valid]
    if pred_vec.numel() < (2 * min_pairs):
        return pred_depth.new_tensor(0.0)

    pair_n = min(int(sample_n), pred_vec.numel() // 2)
    if pair_n < min_pairs:
        return pred_depth.new_tensor(0.0)

    idx = torch.randperm(pred_vec.numel(), device=pred_vec.device)[: 2 * pair_n]
    idx0 = idx[:pair_n]
    idx1 = idx[pair_n:]

    mono0 = mono_vec[idx0]
    mono1 = mono_vec[idx1]
    pred0 = pred_vec[idx0]
    pred1 = pred_vec[idx1]

    order_valid = torch.abs(mono0 - mono1) > mono_eps
    if int(order_valid.sum().item()) < min_pairs:
        return pred_depth.new_tensor(0.0)

    mono0 = mono0[order_valid]
    mono1 = mono1[order_valid]
    pred0 = pred0[order_valid]
    pred1 = pred1[order_valid]

    mono0_closer = mono0 < mono1
    violation = torch.where(mono0_closer, pred0 - pred1, pred1 - pred0)
    return F.relu(violation).mean()


def get_multi_stage_losses(loss_args, depth_types, outputs, depth_gt_ms, mask_ms,
                           depth_interval, inverse_depth):
    depth_loss_weights = loss_args['dlossw']
    global_rel_cfg = loss_args.get("global_relative_depth_loss", {})
    if isinstance(global_rel_cfg, bool):
        global_rel_cfg = {"enabled": global_rel_cfg}
    global_rel_enabled = bool(global_rel_cfg.get("enabled", False))
    global_rel_weight = float(global_rel_cfg.get("weight", 0.01))
    global_rel_stage_set = set()
    for s in global_rel_cfg.get("stages", [4]):
        try:
            si = int(s)
        except (TypeError, ValueError):
            continue
        if 1 <= si <= 4:
            global_rel_stage_set.add(si)
    global_rel_sample_n = max(1, int(global_rel_cfg.get("sample_n", 4096)))
    global_rel_min_pairs = max(1, int(global_rel_cfg.get("min_pairs", 128)))
    global_rel_min_depth = float(global_rel_cfg.get("min_depth", 1e-3))
    global_rel_mono_eps = float(global_rel_cfg.get("mono_eps", 1e-4))
    global_rel_use_conf_mask = bool(global_rel_cfg.get("use_confidence_mask", True))
    global_rel_conf_thresholds = global_rel_cfg.get("confidence_thresholds", [0.15, 0.15, 0.3, 0.3])
    if not isinstance(global_rel_conf_thresholds, (list, tuple)):
        global_rel_conf_thresholds = [0.15, 0.15, 0.3, 0.3]
    loss_dict = {}
    stage_keys = [k for k in ['stage1', 'stage2', 'stage3', 'stage4'] if k in outputs]
    assert len(stage_keys) == len(depth_types)
    for (stage_inputs, stage_key) in [(outputs[k], k) for k in stage_keys]:
        stage_idx = int(stage_key.replace("stage", "")) - 1
        depth_type = depth_types[stage_idx]
        assert depth_type in ("ce", "reg")
        if depth_type == "ce":
            depth_gt = depth_gt_ms[stage_key]
            depth_values = stage_inputs["depth_values"]  # (b, d, h, w)
            prob_volume_pre = stage_inputs["prob_volume_pre"].to(torch.float32)
            mask = mask_ms[stage_key]
            mask = (mask > 0.5).to(torch.float32)

            depth_gt = depth_gt.unsqueeze(1)
            depth_gt_volume = depth_gt.expand_as(depth_values)  # (b, d, h, w)
            # Inverse depth reverses the hypothesis ordering.
            if inverse_depth:  # and stage_key == 'stage1' all stages should be reversed
                depth_values = torch.flip(depth_values, dims=[1])
                prob_volume_pre = torch.flip(prob_volume_pre, dims=[1])
            intervals = torch.abs(depth_values[:, 1:] - depth_values[:, :-1]) / 2  # [b,d-1,h,w]
            intervals = torch.cat([intervals, intervals[:, -1:]], dim=1)  # [b,d,h,w]
            min_depth_values = depth_values[:, 0:1] - intervals[:, 0:1, ]
            max_depth_values = depth_values[:, -1:] + intervals[:, -1:]
            depth_values_right = depth_values + intervals
            out_of_range_left = (depth_gt < min_depth_values).to(torch.float32)
            out_of_range_right = (depth_gt > max_depth_values).to(torch.float32)
            out_of_range_mask = torch.clamp(out_of_range_left + out_of_range_right, 0, 1)
            in_range_mask = 1 - out_of_range_mask
            final_mask = in_range_mask.squeeze(1) * mask
            gt_index_volume = (depth_values_right <= depth_gt_volume).to(torch.float32).sum(dim=1, keepdims=True).to(torch.long)
            gt_index_volume = torch.clamp_max(gt_index_volume, max=depth_values.shape[1] - 1).squeeze(1)

            # mask:[B,H,W], prob:[B,D,H,W], gtd:[B,H,W]
            final_mask = final_mask.to(torch.bool)
            gt_index_volume = gt_index_volume[final_mask]  # [N,]
            prob_volume_pre = prob_volume_pre.permute(0, 2, 3, 1)[final_mask, :]  # [B,H,W,D]->[N,D]
            depth_loss = F.cross_entropy(prob_volume_pre, gt_index_volume, reduction='mean')
            # depth_loss = torch.sum(depth_loss * final_mask) / (torch.sum(final_mask) + 1e-6)
            loss_dict[stage_key] = depth_loss_weights[stage_idx] * depth_loss
        elif depth_type == 'reg':
            depth_values = stage_inputs["depth_values"]
            depth_est = stage_inputs["depth"] / depth_interval[:,None,None]
            depth_gt = depth_gt_ms[stage_key] / depth_interval[:,None,None]
            log_var = stage_inputs.get("log_var", None)
            mask = mask_ms[stage_key]
            mask = mask > 0.5
            clip_func = loss_args.get("clip_func", None)
            if clip_func == 'dynamic':
                if inverse_depth:
                    depth_values = torch.flip(depth_values, dims=[1]) # [b,d,h,w]
                depth_range = (depth_values[:,-1] - depth_values[:,0]) / depth_interval[:,None,None] # [b,h,w]
                depth_range = depth_range[mask]
            else:
                depth_range = None

            if log_var is None:
                if clip_func == 'dynamic':
                    depth_loss = F.smooth_l1_loss(depth_est[mask], depth_gt[mask], reduction='none')
                    depth_loss = torch.clamp_max(depth_loss, depth_range)
                    depth_loss = depth_loss.mean()
                else:
                    depth_loss = F.smooth_l1_loss(depth_est[mask], depth_gt[mask], reduction='mean')
            else:
                l1_err = F.l1_loss(depth_est[mask], depth_gt[mask], reduction='none')
                if clip_func == 'dynamic':
                    l1_err = torch.clamp_max(l1_err, depth_range)
                l1_loss = l1_err.mean()
                uncert_loss = l1_err * (-log_var[mask]).exp() + log_var[mask] * loss_args.get("logvar_weight", 0.1)
                valid_uncert_mask = torch.isfinite(uncert_loss)
                uncert_loss = uncert_loss[valid_uncert_mask].mean()
                depth_loss = l1_loss + uncert_loss
                loss_dict[stage_key + "_uncertainty"] = depth_loss_weights[stage_idx] * uncert_loss

            loss_dict[stage_key] = depth_loss_weights[stage_idx] * depth_loss
        else:
            raise NotImplementedError(f"Not implemented {depth_type}")

        if global_rel_enabled and ((stage_idx + 1) in global_rel_stage_set):
            pred_depth = stage_inputs.get("depth", None)
            mono_depth = stage_inputs.get("mono_depth", None)
            if pred_depth is not None and mono_depth is not None:
                valid_mask = (mask_ms[stage_key] > 0.5)
                if global_rel_use_conf_mask:
                    conf = stage_inputs.get("photometric_confidence", None)
                    if conf is not None:
                        if stage_idx < len(global_rel_conf_thresholds):
                            conf_th = float(global_rel_conf_thresholds[stage_idx])
                        else:
                            conf_th = float(global_rel_conf_thresholds[-1])
                        valid_mask = valid_mask | (conf > conf_th)
                rel_loss = global_relative_depth_ranking_loss(
                    pred_depth=pred_depth,
                    mono_depth=mono_depth,
                    valid_mask=valid_mask,
                    sample_n=global_rel_sample_n,
                    min_pairs=global_rel_min_pairs,
                    min_depth=global_rel_min_depth,
                    mono_eps=global_rel_mono_eps,
                )
                loss_dict[f"{stage_key}_global_rel"] = global_rel_weight * rel_loss

    return loss_dict


def get_loss(loss_arg, depth_type, outputs, depth_gt_ms_tmp, mask_ms_tmp, depth_interval, inverse_depth):
    if depth_type == 're':
        loss_dict = reg_loss(outputs, depth_gt_ms_tmp, mask_ms_tmp,
                             dlossw=[1, 1, 1, 1], depth_interval=depth_interval)
    elif depth_type == 'ce':
        loss_dict = ce_loss(outputs, depth_gt_ms_tmp, mask_ms_tmp, dlossw=[1, 1, 1, 1],
                            focal=loss_arg['focal'], gamma=loss_arg['gamma'], inverse_depth=inverse_depth,
                            )
    else:
        raise NotImplementedError(f"Unknown loss {depth_type}")

    return loss_dict


def simple_loss(outputs, depth_gt_ms, mask_ms):
    depth_est = outputs["depth"]
    depth_gt = depth_gt_ms
    mask = mask_ms
    mask = mask > 0.5
    depth_loss = F.smooth_l1_loss(depth_est[mask], depth_gt[mask], reduction='mean')

    return depth_loss


def reg_loss(inputs, depth_gt_ms, mask_ms, dlossw, depth_interval):
    loss_dict = {}
    depth_interval = depth_interval.unsqueeze(-1).unsqueeze(-1)
    depth_loss_weights = dlossw

    for (stage_inputs, stage_key) in [(inputs[k], k) for k in ['stage1', 'stage2', 'stage3', 'stage4']]:
        if stage_key not in depth_gt_ms:
            continue
        depth_est = stage_inputs["depth"] / depth_interval
        depth_gt = depth_gt_ms[stage_key] / depth_interval
        mask = mask_ms[stage_key]
        mask = mask > 0.5

        depth_loss = F.smooth_l1_loss(depth_est[mask], depth_gt[mask], reduction='mean')

        if depth_loss_weights is not None:
            stage_idx = int(stage_key.replace("stage", "")) - 1
            loss_dict[stage_key] = depth_loss_weights[stage_idx] * depth_loss
        else:
            loss_dict[stage_key] = depth_loss

    return loss_dict


def ce_loss(inputs, depth_gt_ms, mask_ms, dlossw, focal=False, gamma=0.0, inverse_depth=True):
    depth_loss_weights = dlossw

    loss_dict = {}
    stage_keys = [k for k in ['stage1', 'stage2', 'stage3', 'stage4'] if k in inputs]
    for (stage_inputs, stage_key) in [(inputs[k], k) for k in stage_keys]:
        depth_gt = depth_gt_ms[stage_key]
        depth_values = stage_inputs["depth_values"]  # (b, d, h, w)
        prob_volume_pre = stage_inputs["prob_volume_pre"].to(torch.float32)
        mask = mask_ms[stage_key]
        mask = (mask > 0.5).to(torch.float32)

        depth_gt = depth_gt.unsqueeze(1)
        depth_gt_volume = depth_gt.expand_as(depth_values)  # (b, d, h, w)
        # Inverse depth reverses the hypothesis ordering.
        if inverse_depth:  # and stage_key == 'stage1' all stages should be reversed
            depth_values = torch.flip(depth_values, dims=[1])
            prob_volume_pre = torch.flip(prob_volume_pre, dims=[1])
        intervals = torch.abs(depth_values[:, 1:] - depth_values[:, :-1]) / 2  # [b,d-1,h,w]
        intervals = torch.cat([intervals, intervals[:, -1:]], dim=1)  # [b,d,h,w]
        min_depth_values = depth_values[:, 0:1] - intervals[:, 0:1, ]
        max_depth_values = depth_values[:, -1:] + intervals[:, -1:]
        depth_values_right = depth_values + intervals
        out_of_range_left = (depth_gt < min_depth_values).to(torch.float32)
        out_of_range_right = (depth_gt > max_depth_values).to(torch.float32)
        out_of_range_mask = torch.clamp(out_of_range_left + out_of_range_right, 0, 1)
        in_range_mask = 1 - out_of_range_mask
        final_mask = in_range_mask.squeeze(1) * mask
        gt_index_volume = (depth_values_right <= depth_gt_volume).to(torch.float32).sum(dim=1, keepdims=True).to(torch.long)
        gt_index_volume = torch.clamp_max(gt_index_volume, max=depth_values.shape[1] - 1).squeeze(1)

        # mask:[B,H,W], prob:[B,D,H,W], gtd:[B,H,W]

        final_mask = final_mask.to(torch.bool)
        gt_index_volume = gt_index_volume[final_mask]  # [N,]
        prob_volume_pre = prob_volume_pre.permute(0, 2, 3, 1)[final_mask, :]  # [B,H,W,D]->[N,D]
        depth_loss = F.cross_entropy(prob_volume_pre, gt_index_volume, reduction='mean')
        # depth_loss = torch.sum(depth_loss * final_mask) / (torch.sum(final_mask) + 1e-6)

    if depth_loss_weights is not None:
        stage_idx = int(stage_key.replace("stage", "")) - 1
        loss_dict[stage_key] = depth_loss_weights[stage_idx] * depth_loss
    else:
        loss_dict[stage_key] = depth_loss

    return loss_dict




def sigmoid(x, base=2.71828):
    return 1 / (1 + torch.pow(base, -x))
