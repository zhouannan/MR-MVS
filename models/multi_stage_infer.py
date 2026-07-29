from collections import OrderedDict

import numpy as np
import torch


def _as_int(value):
    if isinstance(value, torch.Tensor):
        return int(value.reshape(-1)[0].item())
    if isinstance(value, np.ndarray):
        return int(value.reshape(-1)[0])
    return int(value)


class MultiStageInferencer:
    """Chunked four-stage inference followed by final-stage refinement."""

    def __init__(
        self,
        model,
        device,
        num_stages=4,
        tmp=None,
        enable_pair_cache=True,
        max_pair_cache_size=128,
        profile_timing=False,
        amp_dtype=None,
    ):
        self.model = model
        self.device = device
        self.num_stages = int(num_stages)
        self.tmp = tmp or [5.0, 5.0, 5.0, 1.0]
        self.enable_pair_cache = bool(enable_pair_cache)
        self.max_pair_cache_size = max(0, int(max_pair_cache_size))
        self.profile_timing = bool(profile_timing)
        self.amp_dtype = amp_dtype
        self._pair_cache = OrderedDict()
        self.timing = {
            "forward_s": 0.0,
            "forward_count": 0,
            "refine_s": 0.0,
            "refine_count": 0,
        }

    def _time_cuda_call(self, name, function):
        if not self.profile_timing or not torch.cuda.is_available():
            return function()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        result = function()
        end.record()
        end.synchronize()
        self.timing[f"{name}_s"] += start.elapsed_time(end) / 1000.0
        self.timing[f"{name}_count"] += 1
        return result

    def reset_timing(self):
        self.timing.update(
            {
                "forward_s": 0.0,
                "forward_count": 0,
                "refine_s": 0.0,
                "refine_count": 0,
            }
        )

    def timing_summary(self, output_count=0):
        output_count = max(1, int(output_count))
        forward_count = max(1, int(self.timing["forward_count"]))
        refine_count = max(1, int(self.timing["refine_count"]))
        total = self.timing["forward_s"] + self.timing["refine_s"]
        return {
            **self.timing,
            "forward_avg_s": self.timing["forward_s"] / forward_count,
            "refine_avg_s": self.timing["refine_s"] / refine_count,
            "total_profile_s": total,
            "profile_per_depth_s": total / output_count,
            "output_count": output_count,
        }

    def clear_cache(self):
        self._pair_cache.clear()

    def _cache_get(self, key):
        if not self.enable_pair_cache or self.max_pair_cache_size <= 0:
            return None
        value = self._pair_cache.get(key)
        if value is not None:
            self._pair_cache.move_to_end(key)
        return value

    def _cache_put(self, key, value):
        if not self.enable_pair_cache or self.max_pair_cache_size <= 0:
            return
        self._pair_cache[key] = value
        self._pair_cache.move_to_end(key)
        while len(self._pair_cache) > self.max_pair_cache_size:
            self._pair_cache.popitem(last=False)

    def _to_device(self, value):
        if isinstance(value, torch.Tensor):
            return value.to(self.device, non_blocking=True)
        return torch.as_tensor(value, device=self.device)

    def _to_device_dict(self, values):
        result = {}
        for key, value in values.items():
            if isinstance(value, list):
                result[key] = [self._to_device(item) for item in value]
            else:
                result[key] = self._to_device(value)
        return result

    @staticmethod
    def _normalize_pairs(pairs):
        normalized = []
        for ref, sources in pairs:
            normalized.append(
                (
                    _as_int(ref),
                    [_as_int(source) for source in sources],
                )
            )
        return normalized

    def _prepare_pair_inputs(
        self,
        imgs_cpu,
        projections,
        mono_depths,
        depth_values,
        ref_local,
        source_locals,
        projection_source_positions,
    ):
        pair_locals = [ref_local] + source_locals
        imgs = self._to_device(imgs_cpu[:, pair_locals])
        projection_indices = [0] + [
            int(position) + 1
            for position in projection_source_positions
        ]
        selected_projections = {
            stage: values[:, projection_indices]
            for stage, values in projections[ref_local].items()
        }
        cameras = self._to_device_dict(selected_projections)
        mono = None
        if mono_depths is not None and mono_depths[ref_local] is not None:
            mono = self._to_device_dict(mono_depths[ref_local])
        depth_range = self._to_device(depth_values[ref_local])
        return imgs, cameras, mono, depth_range

    def _infer_pair(
        self,
        *,
        namespace,
        imgs_cpu,
        projections,
        mono_depths,
        depth_values,
        view_local,
        ref_view,
        source_views,
        keep_visibility,
    ):
        if ref_view not in view_local:
            return None
        valid_entries = [
            (position, source)
            for position, source in enumerate(source_views)
            if source in view_local
        ]
        source_positions = [position for position, _ in valid_entries]
        valid_sources = [source for _, source in valid_entries]
        cache_key = (
            str(namespace),
            int(ref_view),
            tuple(valid_sources),
            self.num_stages,
        )
        cached = self._cache_get(cache_key)
        if cached is not None and (
            not keep_visibility or "pre_output" in cached
        ):
            return cached

        ref_local = view_local[ref_view]
        source_locals = [view_local[source] for source in valid_sources]
        imgs, cameras, mono, depth_range = self._prepare_pair_inputs(
            imgs_cpu,
            projections,
            mono_depths,
            depth_values,
            ref_local,
            source_locals,
            source_positions,
        )

        def forward():
            with torch.autocast(
                device_type="cuda",
                dtype=self.amp_dtype or torch.float32,
                enabled=self.amp_dtype is not None,
            ):
                return self.model(
                    imgs,
                    cameras,
                    depth_range,
                    mono_depth=mono,
                    tmp=self.tmp,
                )

        outputs = self._time_cuda_call("forward", forward)
        final_stage = outputs[f"stage{self.num_stages}"]
        result = {
            "depth": final_stage["depth"].detach().cpu(),
            "confidence": outputs["photometric_confidence"].detach().cpu(),
            "stage_4_confidence": final_stage[
                "photometric_confidence"
            ].detach().cpu(),
        }
        if keep_visibility:
            result["pre_output"] = {
                "vis_list": [
                    weight.detach().cpu()
                    for weight in final_stage.get("vis_list", [])
                ],
                "source_views": tuple(valid_sources),
            }

        # Visibility maps are large and are only useful in the current chunk.
        self._cache_put(
            cache_key,
            {
                key: value
                for key, value in result.items()
                if key != "pre_output"
            },
        )
        return result

    def _refine(
        self,
        *,
        imgs_cpu,
        projections,
        mono_depths,
        depth_values,
        view_local,
        ref_view,
        source_views,
        depth_map,
        pre_output,
    ):
        valid_entries = [
            (position, source)
            for position, source in enumerate(source_views)
            if source in depth_map and source in view_local
        ]
        source_positions = [position for position, _ in valid_entries]
        valid_sources = [source for _, source in valid_entries]
        if not valid_sources or ref_view not in depth_map:
            return depth_map[ref_view]

        ref_local = view_local[ref_view]
        source_locals = [view_local[source] for source in valid_sources]
        imgs, cameras, _, depth_range = self._prepare_pair_inputs(
            imgs_cpu,
            projections,
            mono_depths,
            depth_values,
            ref_local,
            source_locals,
            source_positions,
        )
        depths = torch.cat(
            [depth_map[ref_view]]
            + [depth_map[source] for source in valid_sources],
            dim=0,
        ).to(self.device, non_blocking=True)
        visibility_sources = list(
            pre_output.get("source_views", source_views)
        )
        visibility_index = {
            int(source): index
            for index, source in enumerate(visibility_sources)
        }
        selected_visibility = [
            pre_output["vis_list"][visibility_index[source]]
            for source in valid_sources
            if source in visibility_index
        ]
        if len(selected_visibility) != len(valid_sources):
            raise RuntimeError(
                "Refinement visibility/source mismatch for reference "
                f"{ref_view}: {len(selected_visibility)} visibility maps "
                f"for {len(valid_sources)} source depths"
            )
        visibility = self._to_device_dict(
            {"vis_list": selected_visibility}
        )

        refined = self._time_cuda_call(
            "refine",
            lambda: self.model.refine_stage_depth(
                imgs=imgs,
                proj_matrices=cameras,
                depths=depths,
                pre_output=visibility,
                depth_values=depth_range,
                stage_idx=self.num_stages - 1,
            ),
        )
        return refined.detach().cpu()

    @torch.no_grad()
    def run(
        self,
        imgs_cpu,
        projections,
        mono_depths,
        depth_values,
        pairs,
        view_ids,
        view_num,
        cache_namespace=None,
    ):
        pairs = self._normalize_pairs(pairs)
        view_ids = [_as_int(view_id) for view_id in view_ids]
        view_local = {
            view_id: index for index, view_id in enumerate(view_ids)
        }
        main_pairs = pairs[: max(0, min(int(view_num), len(pairs)))]
        main_refs = {ref for ref, _ in main_pairs}

        depth_map = {}
        confidence_map = {}
        stage_confidence_map = {}
        visibility_map = {}
        for pair_index, (ref, sources) in enumerate(pairs, start=1):
            if len(pairs) >= 10 and (
                pair_index == 1
                or pair_index % 10 == 0
                or pair_index == len(pairs)
            ):
                print(
                    f"[MR-MVS] forward pair {pair_index}/{len(pairs)}",
                    flush=True,
                )
            result = self._infer_pair(
                namespace=cache_namespace,
                imgs_cpu=imgs_cpu,
                projections=projections,
                mono_depths=mono_depths,
                depth_values=depth_values,
                view_local=view_local,
                ref_view=ref,
                source_views=sources,
                keep_visibility=ref in main_refs,
            )
            if result is None:
                continue
            depth_map[ref] = result["depth"]
            confidence_map[ref] = result["confidence"]
            stage_confidence_map[ref] = result["stage_4_confidence"]
            if "pre_output" in result:
                visibility_map[ref] = result["pre_output"]

        refined_count = 0
        for ref, sources in main_pairs:
            if ref not in depth_map or ref not in visibility_map:
                continue
            depth_map[ref] = self._refine(
                imgs_cpu=imgs_cpu,
                projections=projections,
                mono_depths=mono_depths,
                depth_values=depth_values,
                view_local=view_local,
                ref_view=ref,
                source_views=sources,
                depth_map=depth_map,
                pre_output=visibility_map[ref],
            )
            refined_count += 1
        print(
            f"[MR-MVS] final-stage refine: "
            f"{refined_count}/{len(main_pairs)} references",
            flush=True,
        )

        return {
            ref: {
                "depth": depth_map[ref],
                "confidence": confidence_map[ref],
                "stage_4_confidence": stage_confidence_map[ref],
            }
            for ref, _ in main_pairs
            if ref in depth_map
        }
