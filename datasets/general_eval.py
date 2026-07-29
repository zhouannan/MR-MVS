import os

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from datasets.data_io import read_pfm


class MVSDataset(Dataset):
    """Evaluation dataset with one reference view and its source views.

    Camera files store quarter-resolution intrinsics, following the original
    MVSFormer++ Tanks-and-Temples preprocessing. Images, monocular depths, and
    cameras are all mapped to the same requested pixel grid here.
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
        super().__init__()
        if mode != "test":
            raise ValueError("MVSDataset is an evaluation-only dataset")

        self.datapath = datapath
        self.listfile = listfile
        self.mode = mode
        self.nviews = int(nviews)
        self.ndepths = int(ndepths)
        self.mono_depths_path = mono_depths_path
        self.max_h = int(kwargs["max_h"])
        self.max_w = int(kwargs["max_w"])
        self.dataset = kwargs.get("dataset", "tt")
        self.stage3 = bool(kwargs.get("stage3", False))
        self.use_short_range = bool(kwargs.get("use_short_range", False))
        self.transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.485, 0.456, 0.406),
                    (0.229, 0.224, 0.225),
                ),
            ]
        )

        self.scans = self._read_scans(listfile)
        self.interval_scales = {
            scan: (
                float(interval_scale)
                if isinstance(interval_scale, (int, float))
                else float(interval_scale[scan])
            )
            for scan in self.scans
        }
        self.metas = self._build_list()

    @staticmethod
    def _read_scans(listfile):
        if isinstance(listfile, (list, tuple)):
            return [str(scan).strip() for scan in listfile if str(scan).strip()]
        with open(listfile, "r", encoding="utf-8") as handle:
            return [line.strip() for line in handle if line.strip()]

    def _build_list(self):
        metas = []
        for scan in self.scans:
            pair_file = os.path.join(self.datapath, scan, "pair.txt")
            if not os.path.isfile(pair_file):
                pair_file = os.path.join(
                    self.datapath, scan, "new_pair.txt"
                )
            with open(pair_file, "r", encoding="utf-8") as handle:
                num_viewpoints = int(handle.readline())
                for _ in range(num_viewpoints):
                    ref_view = int(handle.readline().strip())
                    tokens = handle.readline().strip().split()
                    src_views = [int(value) for value in tokens[1::2]]
                    if not src_views:
                        continue
                    if len(src_views) < self.nviews - 1:
                        src_views.extend(
                            [src_views[0]] * (self.nviews - 1 - len(src_views))
                        )
                    metas.append(
                        (scan, ref_view, src_views[: self.nviews - 1])
                    )
        print(
            f"dataset {self.mode} metas: {len(metas)} "
            f"interval_scale: {self.interval_scales}"
        )
        return metas

    def __len__(self):
        return len(self.metas)

    def _image_path(self, scan, view_id):
        for directory in ("images", "images_test"):
            path = os.path.join(
                self.datapath, scan, directory, f"{view_id:08d}.jpg"
            )
            if os.path.isfile(path):
                return path
        raise FileNotFoundError(
            f"No image found for {scan}/{view_id:08d}"
        )

    def _camera_path(self, scan, view_id):
        if self.dataset == "tt" and self.use_short_range:
            path = os.path.join(
                self.datapath,
                "short_range_cameras",
                f"cams_{scan.lower()}",
                f"{view_id:08d}_cam.txt",
            )
            if os.path.isfile(path):
                return path

        prefer_cams = os.environ.get(
            "MVSFORMER_TT_PREFER_CAMS", "0"
        ) == "1"
        directories = (
            ("cams", "cams_1")
            if prefer_cams or self.dataset != "tt"
            else ("cams_1", "cams")
        )
        for directory in directories:
            path = os.path.join(
                self.datapath,
                scan,
                directory,
                f"{view_id:08d}_cam.txt",
            )
            if os.path.isfile(path):
                return path
        raise FileNotFoundError(
            f"No camera found for {scan}/{view_id:08d}"
        )

    def _mono_path(self, scan, view_id):
        if not self.mono_depths_path:
            return None
        name = f"{view_id:08d}.npy"
        candidates = (
            os.path.join(self.mono_depths_path, name),
            os.path.join(
                self.mono_depths_path, scan, "mono_depth", name
            ),
            os.path.join(self.mono_depths_path, scan, name),
            os.path.join(self.datapath, scan, "mono_depth", name),
        )
        for path in candidates:
            if os.path.isfile(path):
                return path
        raise FileNotFoundError(
            f"No monocular depth found for {scan}/{view_id:08d}; "
            f"checked: {candidates}"
        )

    def read_img(self, filename):
        image = np.asarray(Image.open(filename).convert("RGB"))
        if self.dataset == "tt":
            image = np.pad(image, ((4, 4), (0, 0), (0, 0)), mode="edge")
        return image

    def read_cam_file(self, filename, interval_scale):
        with open(filename, "r", encoding="utf-8") as handle:
            lines = [line.rstrip() for line in handle]

        extrinsics = np.fromstring(
            " ".join(lines[1:5]), dtype=np.float32, sep=" "
        ).reshape(4, 4)
        intrinsics = np.fromstring(
            " ".join(lines[7:10]), dtype=np.float32, sep=" "
        ).reshape(3, 3)

        if self.dataset == "tt":
            intrinsics[1, 2] += 4.0
        intrinsics[:2] /= 4.0

        parts = lines[11].split()
        depth_min = float(parts[0])
        if "cams_1" in filename and self.dataset != "tt":
            depth_interval = 2.5
        else:
            depth_interval = float(parts[1])

        if self.dataset == "tt" and "cams_1" in filename and len(parts) == 2:
            depth_max = float(parts[1])
            depth_interval = (depth_max - depth_min) / self.ndepths
        elif len(parts) >= 3:
            depth_max = depth_min + int(float(parts[2])) * depth_interval
            depth_interval = (depth_max - depth_min) / self.ndepths

        if self.dataset == "eth3d":
            depth_interval = (float(parts[1]) - depth_min) / self.ndepths

        return (
            intrinsics,
            extrinsics,
            depth_min,
            depth_interval * float(interval_scale),
        )

    def resize_image_and_camera(self, image, intrinsics):
        source_h, source_w = image.shape[:2]
        scale_w = self.max_w / float(source_w)
        scale_h = self.max_h / float(source_h)
        intrinsics = intrinsics.copy()
        intrinsics[0] *= scale_w
        intrinsics[1] *= scale_h
        image = cv2.resize(
            image, (self.max_w, self.max_h), interpolation=cv2.INTER_AREA
        )
        return image, intrinsics

    def prepare_mono_depth(self, depth, source_padded_hw):
        depth = np.asarray(depth, dtype=np.float32)
        padded_h, padded_w = source_padded_hw
        if self.dataset == "tt" and depth.shape == (padded_h - 8, padded_w):
            depth = np.pad(depth, ((4, 4), (0, 0)), mode="edge")
        if depth.shape != (self.max_h, self.max_w):
            depth = cv2.resize(
                depth,
                (self.max_w, self.max_h),
                interpolation=cv2.INTER_LINEAR,
            )
        return depth

    @staticmethod
    def generate_stage_depth(depth):
        height, width = depth.shape
        return {
            "stage1": cv2.resize(
                depth,
                (width // 8, height // 8),
                interpolation=cv2.INTER_NEAREST,
            ),
            "stage2": cv2.resize(
                depth,
                (width // 4, height // 4),
                interpolation=cv2.INTER_NEAREST,
            ),
            "stage3": cv2.resize(
                depth,
                (width // 2, height // 2),
                interpolation=cv2.INTER_NEAREST,
            ),
            "stage4": depth,
        }

    def make_proj_ms(self, projections):
        stage1 = projections.copy()
        stage1[:, 1, :2] *= 0.5
        stage2 = projections.copy()
        stage3 = projections.copy()
        stage3[:, 1, :2] *= 2.0
        stage4 = projections.copy()
        stage4[:, 1, :2] *= 4.0
        result = {
            "stage1": stage1,
            "stage2": stage2,
            "stage3": stage3,
            "stage4": stage4,
        }
        if self.stage3:
            result = {
                "stage1": result["stage2"],
                "stage2": result["stage3"],
                "stage3": result["stage4"],
            }
        return result

    def _read_dtu_ground_truth(self, scan, view_id):
        training_root = os.path.dirname(self.datapath.rstrip("/\\"))
        depth_root = os.path.join(
            training_root, "mvs_training", "Depths_raw", scan
        )
        depth_path = os.path.join(
            depth_root, f"depth_map_{view_id:04d}.pfm"
        )
        mask_path = os.path.join(
            depth_root, f"depth_visual_{view_id:04d}.png"
        )
        if not os.path.isfile(depth_path) or not os.path.isfile(mask_path):
            depth = np.zeros((self.max_h, self.max_w), dtype=np.float32)
            mask = np.ones_like(depth)
        else:
            depth = np.asarray(read_pfm(depth_path)[0], dtype=np.float32)
            depth = cv2.resize(
                depth,
                (self.max_w, self.max_h),
                interpolation=cv2.INTER_NEAREST,
            )
            mask = np.asarray(Image.open(mask_path), dtype=np.float32)
            mask = cv2.resize(
                (mask > 10).astype(np.float32),
                (self.max_w, self.max_h),
                interpolation=cv2.INTER_NEAREST,
            )
        return self.generate_stage_depth(depth), self.generate_stage_depth(mask)

    def __getitem__(self, index):
        scan, ref_view, src_views = self.metas[index]
        view_ids = [ref_view] + src_views
        images = []
        projections = []
        depth_values = None
        mono_depth = None

        for view_index, view_id in enumerate(view_ids):
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
            images.append(
                self.transforms(Image.fromarray(image))
            )

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
            projections.append(projection)

            if view_index == 0:
                depth_values = (
                    depth_min
                    + np.arange(self.ndepths, dtype=np.float32)
                    * depth_interval
                )
                mono_path = self._mono_path(scan, ref_view)
                if mono_path is not None:
                    mono = self.prepare_mono_depth(
                        np.load(mono_path), padded_hw
                    )
                    mono_depth = self.generate_stage_depth(mono)

        sample = {
            "scan": scan,
            "imgs": torch.stack(images),
            "proj_matrices": self.make_proj_ms(
                np.stack(projections)
            ),
            "depth_values": depth_values,
            "filename": (
                scan + "/{}/" + f"{ref_view:08d}" + "{}"
            ),
        }
        if mono_depth is not None:
            sample["mono_depth"] = mono_depth
        if self.dataset == "dtu":
            depth, mask = self._read_dtu_ground_truth(scan, ref_view)
            sample["depth"] = depth
            sample["mask"] = mask
        return sample
