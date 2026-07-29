from collections import OrderedDict

import numpy as np
import torch
from PIL import Image

from datasets.general_eval import MVSDataset


class MVSDatasetAll(MVSDataset):
    """Load a chunk of reference views plus the views needed for refinement.

    A main reference requires depths for its source views. Those source depths
    are inferred from their own pairs, so each chunk contains main references,
    their one-hop sources, and the corresponding two-hop image dependencies.
    """

    def __init__(
        self,
        datapath,
        listfile,
        mode,
        nviews,
        mono_depths_path=None,
        ndepths=192,
        interval_scale=1.06,
        **kwargs,
    ):
        if mode != "multi_stage":
            raise ValueError("MVSDatasetAll requires mode='multi_stage'")
        super().__init__(
            datapath=datapath,
            listfile=listfile,
            mode="test",
            nviews=nviews,
            mono_depths_path=mono_depths_path,
            ndepths=ndepths,
            interval_scale=interval_scale,
            **kwargs,
        )
        self.mode = mode
        self.chunk_size = max(1, int(kwargs.get("chunk_size", 50)))
        self.view_cache_size = max(
            0, int(kwargs.get("view_cache_size", 32))
        )
        self._view_cache = OrderedDict()

        self.scan_pairs = {scan: [] for scan in self.scans}
        for scan, ref_view, src_views in self.metas:
            self.scan_pairs[scan].append((ref_view, src_views))
        self.scan_pair_maps = {
            scan: {ref: srcs for ref, srcs in pairs}
            for scan, pairs in self.scan_pairs.items()
        }
        self.chunks = []
        for scan in self.scans:
            for start in range(0, len(self.scan_pairs[scan]), self.chunk_size):
                self.chunks.append((scan, start))

    def __len__(self):
        return len(self.chunks)

    def _cache_get(self, key):
        value = self._view_cache.get(key)
        if value is not None:
            self._view_cache.move_to_end(key)
        return value

    def _cache_put(self, key, value):
        if self.view_cache_size <= 0:
            return
        self._view_cache[key] = value
        self._view_cache.move_to_end(key)
        while len(self._view_cache) > self.view_cache_size:
            self._view_cache.popitem(last=False)

    def _load_view(self, scan, view_id):
        key = (scan, int(view_id))
        cached = self._cache_get(key)
        if cached is not None:
            return cached

        image = self.read_img(self._image_path(scan, view_id))
        padded_hw = image.shape[:2]
        intrinsics, extrinsics, depth_min, depth_interval = (
            self.read_cam_file(
                self._camera_path(scan, view_id),
                self.interval_scales[scan],
            )
        )
        image, intrinsics = self.resize_image_and_camera(
            image, intrinsics
        )
        image_tensor = self.transforms(Image.fromarray(image))

        projection = np.zeros((2, 4, 4), dtype=np.float32)
        projection[0] = extrinsics
        projection[1, :3, :3] = intrinsics
        projection[1, 3] = np.asarray(
            [
                depth_min,
                depth_interval,
                self.ndepths,
                depth_min + depth_interval * self.ndepths,
            ],
            dtype=np.float32,
        )
        depth_values = (
            depth_min
            + np.arange(self.ndepths, dtype=np.float32) * depth_interval
        )

        mono_depth = None
        mono_path = self._mono_path(scan, view_id)
        if mono_path is not None:
            mono = self.prepare_mono_depth(np.load(mono_path), padded_hw)
            mono_depth = self.generate_stage_depth(mono)

        value = {
            "image": image_tensor,
            "projection": projection,
            "depth_values": depth_values,
            "mono_depth": mono_depth,
        }
        self._cache_put(key, value)
        return value

    @staticmethod
    def _ordered_unique(values):
        seen = set()
        result = []
        for value in values:
            value = int(value)
            if value not in seen:
                seen.add(value)
                result.append(value)
        return result

    def __getitem__(self, index):
        scan, start = self.chunks[index]
        all_pairs = self.scan_pairs[scan]
        pair_map = self.scan_pair_maps[scan]
        main_pairs = all_pairs[start : start + self.chunk_size]

        main_refs = [ref for ref, _ in main_pairs]
        one_hop = self._ordered_unique(
            source for _, sources in main_pairs for source in sources
        )
        auxiliary_refs = [
            view for view in one_hop
            if view not in set(main_refs) and view in pair_map
        ]
        inference_pairs = list(main_pairs) + [
            (ref, pair_map[ref]) for ref in auxiliary_refs
        ]
        dependency_views = self._ordered_unique(
            view
            for ref, sources in inference_pairs
            for view in [ref] + list(sources)
        )

        records = {
            view_id: self._load_view(scan, view_id)
            for view_id in dependency_views
        }
        view_ids = self._ordered_unique(
            [ref for ref, _ in inference_pairs] + dependency_views
        )
        view_local = {
            view_id: local_index
            for local_index, view_id in enumerate(view_ids)
        }

        images = torch.stack(
            [records[view_id]["image"] for view_id in view_ids]
        ).unsqueeze(0)
        projection_sets = []
        mono_depths = []
        depth_values = []
        for view_id in view_ids:
            sources = pair_map.get(view_id, [])
            pair_views = [
                candidate
                for candidate in [view_id] + list(sources)
                if candidate in records
            ]
            projections = np.stack(
                [records[candidate]["projection"] for candidate in pair_views]
            )
            projection_sets.append(
                {
                    stage: torch.from_numpy(values).unsqueeze(0)
                    for stage, values in self.make_proj_ms(
                        projections
                    ).items()
                }
            )
            mono = records[view_id]["mono_depth"]
            if mono is not None:
                mono = {
                    stage: torch.from_numpy(values).unsqueeze(0)
                    for stage, values in mono.items()
                }
            mono_depths.append(mono)
            depth_values.append(
                torch.from_numpy(
                    records[view_id]["depth_values"]
                ).unsqueeze(0)
            )

        # Keep only sources that are present in this chunk's dependency set.
        normalized_pairs = []
        for ref, sources in inference_pairs:
            normalized_pairs.append(
                (
                    int(ref),
                    [
                        int(source)
                        for source in sources
                        if source in view_local
                    ],
                )
            )

        sample = {
            "scan": scan,
            "pairs": normalized_pairs,
            "view_ids": view_ids,
            "imgs_all": images,
            "projs_all": projection_sets,
            "depth_values_all": depth_values,
            "view_num": len(main_pairs),
            "src_num": len(inference_pairs),
        }
        if all(mono is not None for mono in mono_depths):
            sample["mono_all"] = mono_depths
        return sample
