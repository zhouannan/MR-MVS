import io
import os
import shlex
import subprocess
import numpy as np
from struct import pack, unpack
import cv2
from plyfile import PlyData, PlyElement
import misc.fusion as fusion
import torch
from torch.utils.data import Dataset, DataLoader, SequentialSampler
from utils import *
from PIL import Image
from depth2ply import resize_depth_map
import re

def read_pfm(filename):
    file = open(filename, 'rb')

    header = file.readline().decode('utf-8').rstrip()
    if header == 'PF':
        color = True
    elif header == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode('utf-8'))
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().rstrip())
    if scale < 0:  # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>'  # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    file.close()
    return data, scale

def resize_intrinsics(intrinsic,H_old,W_old,H_new,W_new):

    scale_x = W_new / W_old
    scale_y = H_new / H_old

    new_intrinsic = intrinsic.copy()
    new_intrinsic[0, 0] *= scale_x  # fx
    new_intrinsic[1, 1] *= scale_y  # fy
    new_intrinsic[0, 2] *= scale_x  # cx
    new_intrinsic[1, 2] *= scale_y  # cy

    return new_intrinsic

# read an image
def read_img(filename):
    img = Image.open(filename)
    # scale 0~255 to 0~1
    np_img = np.array(img, dtype=np.float32) / 255.
    return np_img

def read_pair_file(filename):
    data = []
    with open(filename) as f:
        num_viewpoint = int(f.readline())
        # 49 viewpoints
        for view_idx in range(num_viewpoint):
            ref_view = int(f.readline().rstrip())
            src_views = [int(x) for x in f.readline().rstrip().split()[1::2]]
            if len(src_views) > 0:
                data.append((ref_view, src_views))
    return data

class TTDataset(Dataset):
    def __init__(
        self,
        input_folder,
        depth_folder,
        conf_choose,
        n_src_views=10,
        load_src_confs=True,
        validated_direct_io=False,
        preload_depths=False,
        preload_fusion_inputs=False,
        allow_camera_rescale=False,
    ):
        super(TTDataset, self).__init__()
        pair_file = os.path.join(input_folder, "pair.txt")
        self.scan_folder = input_folder
        self.pair_data = read_pair_file(pair_file)
        self.n_src_views = n_src_views
        self.conf_choose = conf_choose
        self.depth_folder = depth_folder
        self.load_src_confs = bool(load_src_confs)
        self.validated_direct_io = bool(validated_direct_io)
        self.allow_camera_rescale = bool(allow_camera_rescale)
        self._depth_cache = {}
        self._image_bytes_cache = {}
        self._conf_cache = {}
        self._camera_cache = {}
        view_ids = sorted(
            {ref for ref, _ in self.pair_data}
            | {source for _, sources in self.pair_data for source in sources}
        )
        if preload_depths:
            print(f"[TTDataset] preloading {len(view_ids)} depth maps", flush=True)
            for index, view_id in enumerate(view_ids, start=1):
                path = os.path.join(
                    self.scan_folder,
                    self.depth_folder,
                    '{:0>8}.npy'.format(view_id),
                )
                self._depth_cache[view_id] = np.load(path).astype(np.float32, copy=False)
                if index % 50 == 0 or index == len(view_ids):
                    print(f"[TTDataset] preloaded {index}/{len(view_ids)}", flush=True)
        if preload_fusion_inputs:
            ref_ids = sorted({ref for ref, _ in self.pair_data})
            print(
                f"[TTDataset] preloading {len(view_ids)} cameras and "
                f"{len(ref_ids)} reference images/confidences",
                flush=True,
            )
            for index, view_id in enumerate(view_ids, start=1):
                self._camera_cache[view_id] = read_camera_parameters(
                    os.path.join(
                        self.scan_folder,
                        "cams_test/{:0>8}_cam.txt".format(view_id),
                    )
                )
                if index % 50 == 0 or index == len(view_ids):
                    print(
                        f"[TTDataset] preloaded cameras {index}/{len(view_ids)}",
                        flush=True,
                    )
            for index, view_id in enumerate(ref_ids, start=1):
                image_path = os.path.join(
                    self.scan_folder,
                    "images_test/{:0>8}.jpg".format(view_id),
                )
                with open(image_path, "rb") as handle:
                    self._image_bytes_cache[view_id] = handle.read()
                self._conf_cache[view_id] = np.load(
                    os.path.join(
                        self.scan_folder,
                        self.conf_choose,
                        "{:0>8}.npy".format(view_id),
                    )
                )
                if index % 50 == 0 or index == len(ref_ids):
                    print(
                        f"[TTDataset] preloaded references {index}/{len(ref_ids)}",
                        flush=True,
                    )

    def _load_depth(self, view_id):
        cached = self._depth_cache.get(view_id)
        if cached is not None:
            return cached
        return np.load(
            os.path.join(
                self.scan_folder,
                self.depth_folder,
                '{:0>8}.npy'.format(view_id),
            )
        )

    def _load_ref_image(self, view_id):
        encoded = self._image_bytes_cache.get(view_id)
        if encoded is None:
            return read_img(
                os.path.join(
                    self.scan_folder,
                    "images_test/{:0>8}.jpg".format(view_id),
                )
            )
        with Image.open(io.BytesIO(encoded)) as image:
            return np.array(image, dtype=np.float32) / 255.0

    def _load_confidence(self, view_id):
        cached = self._conf_cache.get(view_id)
        if cached is not None:
            return cached
        return np.load(
            os.path.join(
                self.scan_folder,
                self.conf_choose,
                "{:0>8}.npy".format(view_id),
            )
        )

    def _load_camera(self, view_id):
        cached = self._camera_cache.get(view_id)
        if cached is None:
            cached = read_camera_parameters(
                os.path.join(
                    self.scan_folder,
                    "cams_test/{:0>8}_cam.txt".format(view_id),
                )
            )
        return cached[0].copy(), cached[1].copy()

    def __len__(self):
        return len(self.pair_data)

    def __getitem__(self, idx):
        id_ref, id_srcs = self.pair_data[idx]
        id_srcs = id_srcs[:self.n_src_views]

        # load the reference image
        ref_img = self._load_ref_image(id_ref)
        H_old, W_old = ref_img.shape[:2]
        ref_img = ref_img.transpose([2, 0, 1])
        # load the estimated depth of the reference view
        ref_depth_est = self._load_depth(id_ref)
        ref_depth_est = np.array(ref_depth_est, dtype=np.float32)
        ref_conf = self._load_confidence(id_ref)
        if ref_conf.shape != ref_depth_est.shape:
            raise ValueError(
                f"Reference confidence/depth mismatch for {id_ref}: "
                f"{ref_conf.shape} versus {ref_depth_est.shape}"
            )

        ref_intrinsics, ref_extrinsics = self._load_camera(id_ref)
        if ref_depth_est.shape != (H_old, W_old):
            if not self.allow_camera_rescale:
                raise ValueError(
                    f"Reference image/depth mismatch for {id_ref}: "
                    f"{(H_old, W_old)} versus {ref_depth_est.shape}. "
                    "New MR-MVS outputs must share one pixel grid."
                )
            ref_intrinsics = resize_depth_map(
                ref_depth_est,
                ref_intrinsics,
                H_old,
                W_old,
            )
        ref_cam = np.zeros((2, 4, 4), dtype=np.float32)
        ref_cam[0] = ref_extrinsics
        ref_cam[1, :3, :3] = ref_intrinsics
        ref_cam[1, 3, 3] = 1.0
        if ref_conf.dtype == np.uint8:
            ref_conf = ref_conf / 255.0

        # load the photometric mask of the reference view

        src_depths, src_confs, src_cams = [], [], []
        for ids in id_srcs:
            if (
                not self.validated_direct_io
                and not os.path.exists(
                    os.path.join(
                        self.scan_folder,
                        'cams_test/{:0>8}_cam.txt'.format(ids),
                    )
                )
            ):
                continue
            # the estimated depth of the source view
            src_depth_est = self._load_depth(ids)
            src_intrinsics, src_extrinsics = self._load_camera(ids)
            src_proj = np.zeros((2, 4, 4), dtype=np.float32)
            if not self.validated_direct_io:
                with Image.open(
                    os.path.join(self.scan_folder, 'images_test/{:0>8}.jpg'.format(ids))
                ) as src_image:
                    W_old, H_old = src_image.size
                if src_depth_est.shape != (H_old, W_old):
                    if not self.allow_camera_rescale:
                        raise ValueError(
                            f"Source image/depth mismatch for {ids}: "
                            f"{(H_old, W_old)} versus "
                            f"{src_depth_est.shape}. New MR-MVS outputs "
                            "must share one pixel grid."
                        )
                    src_intrinsics = resize_depth_map(
                        src_depth_est,
                        src_intrinsics,
                        H_old,
                        W_old,
                    )
            src_proj[0] = src_extrinsics
            src_proj[1, :3, :3] = src_intrinsics
            src_proj[1, 3, 3] = 1.0
            src_cams.append(src_proj)
            src_depths.append(np.asarray(src_depth_est, dtype=np.float32))
            if self.load_src_confs:
                src_conf = np.load(os.path.join(self.scan_folder, self.conf_choose,'{:0>8}.npy'.format(ids)))
                if src_conf.dtype == np.uint8:
                    src_conf = src_conf / 255.0
                if src_conf.shape != src_depth_est.shape:
                    raise ValueError(
                        f"Source confidence/depth mismatch for {ids}: "
                        f"{src_conf.shape} versus {src_depth_est.shape}"
                    )
                src_confs.append(src_conf)
        src_depths = np.expand_dims(np.stack(src_depths, axis=0), axis=1)
        src_cams = np.stack(src_cams, axis=0)
        sample = {"ref_depth": np.expand_dims(ref_depth_est, axis=0),
                  "ref_cam": ref_cam,
                  "ref_conf":np.expand_dims(ref_conf, axis=0),
                  "src_depths": src_depths,
                  "src_cams": src_cams,
                  "ref_img": ref_img,
                  "ref_id": id_ref}
        if self.load_src_confs:
            sample["src_confs"] = np.stack(src_confs, axis=0)
        return sample


def collate_single_no_copy(batch):
    if len(batch) != 1:
        raise ValueError("collate_single_no_copy requires batch_size=1")
    collated = {}
    for key, value in batch[0].items():
        if isinstance(value, np.ndarray):
            tensor = torch.from_numpy(value)
        elif torch.is_tensor(value):
            tensor = value
        else:
            tensor = torch.as_tensor(value)
        collated[key] = tensor.unsqueeze(0)
    return collated

# =========================================================
# I/O helpers
# =========================================================

def read_camera_parameters(filename):
    with open(filename) as f:
        lines = f.readlines()
        lines = [line.rstrip() for line in lines]
    # extrinsics: line [1,5), 4x4 matrix
    extrinsics = np.fromstring(' '.join(lines[1:5]), dtype=np.float32, sep=' ').reshape((4, 4))
    # intrinsics: line [7-10), 3x3 matrix
    intrinsics = np.fromstring(' '.join(lines[7:10]), dtype=np.float32, sep=' ').reshape((3, 3))
    # TODO: assume the feature is 1/4 of the original image size
    # intrinsics[:2, :] /= 4
    return intrinsics, extrinsics


PLY_VERTEX_DTYPE = np.dtype(
    [
        ('x', '<f4'), ('y', '<f4'), ('z', '<f4'),
        ('red', 'u1'), ('green', 'u1'), ('blue', 'u1'),
    ]
)


def write_vertex_ply_streaming(plyfilename, views):
    writer = OnlineVertexPlyWriter(plyfilename)
    for points, colors in views.values():
        writer.append(points, colors)
    return writer.close()


class OnlineVertexPlyWriter:
    COUNT_WIDTH = 20

    def __init__(self, plyfilename):
        self.plyfilename = os.fspath(plyfilename)
        self.temporary = self.plyfilename + ".tmp"
        self.handle = open(self.temporary, "w+b")
        prefix = (
            "ply\n"
            "format binary_little_endian 1.0\n"
            "element vertex "
        ).encode("ascii")
        suffix = (
            "\n"
            "property float x\n"
            "property float y\n"
            "property float z\n"
            "property uchar red\n"
            "property uchar green\n"
            "property uchar blue\n"
            "end_header\n"
        ).encode("ascii")
        self.count_offset = len(prefix)
        self.handle.write(prefix)
        self.handle.write(f"{0:{self.COUNT_WIDTH}d}".encode("ascii"))
        self.handle.write(suffix)
        self.vertex_count = 0

    def append(self, points, colors):
        if len(points) != len(colors):
            raise ValueError("Point and color counts must match")
        vertices = np.empty(len(points), dtype=PLY_VERTEX_DTYPE)
        for column, prop in enumerate(("x", "y", "z")):
            vertices[prop] = points[:, column]
        for column, prop in enumerate(("red", "green", "blue")):
            vertices[prop] = colors[:, column]
        vertices.tofile(self.handle)
        self.vertex_count += len(vertices)
        return len(vertices)

    def close(self):
        if self.handle is None:
            return self.vertex_count
        self.handle.seek(self.count_offset)
        self.handle.write(
            f"{self.vertex_count:{self.COUNT_WIDTH}d}".encode("ascii")
        )
        self.handle.close()
        self.handle = None
        os.replace(self.temporary, self.plyfilename)
        return self.vertex_count

    def abort(self):
        if self.handle is not None:
            self.handle.close()
            self.handle = None
        if os.path.exists(self.temporary):
            os.remove(self.temporary)


# =========================================================
# Gipuma .dmb I/O
# =========================================================

def write_gipuma_dmb(path, image):
    image = image.astype(np.float32)
    h, w = image.shape[:2]
    c = 1 if image.ndim == 2 else image.shape[2]

    if image.ndim == 3:
        image = np.transpose(image, (2, 0, 1))

    with open(path, "wb") as f:
        f.write(pack('<i', 1))
        f.write(pack('<i', h))
        f.write(pack('<i', w))
        f.write(pack('<i', c))
        image.tofile(f)


def fake_gipuma_normal(depth_dmb_path, normal_dmb_path):
    with open(depth_dmb_path, "rb") as f:
        _ = unpack('<i', f.read(4))
        h = unpack('<i', f.read(4))[0]
        w = unpack('<i', f.read(4))[0]
        _ = unpack('<i', f.read(4))[0]
        depth = np.fromfile(f, np.float32).reshape((h, w))

    normal = np.ones((h, w, 3), dtype=np.float32)
    normal /= np.linalg.norm(normal, axis=2, keepdims=True)

    mask = (depth > 0).astype(np.float32)[..., None]
    normal *= mask

    write_gipuma_dmb(normal_dmb_path, normal)


# =========================================================
# Camera conversion (MVSNet -> Gipuma)
# =========================================================

def mvsnet_to_gipuma_cam(in_cam_txt, out_cam_P,old_hw,new_hw):
    H_old, W_old = old_hw
    H_new, W_new = new_hw
    intrinsic, extrinsic = read_camera_parameters(in_cam_txt)
    intrinsic = resize_intrinsics(
        intrinsic,
        H_old,
        W_old,
        H_new,
        W_new,
    )
    intrinsic_new = np.zeros((4, 4))
    intrinsic_new[:3, :3] = intrinsic

    intrinsic = intrinsic_new

    projection_matrix = np.matmul(intrinsic, extrinsic)
    projection_matrix = projection_matrix[0:3][:]

    f = open(out_cam_P, "w")
    for i in range(0, 3):
        for j in range(0, 4):
            f.write(str(projection_matrix[i][j]) + ' ')
        f.write('\n')
    f.write('\n')
    f.close()

    return


# =========================================================
# Convert depth_multi_stage -> Gipuma input
# =========================================================

def mvs_depth_to_gipuma(
    dense_folder,
    gipuma_folder,
    depth_folder,
    conf_choose,
    prob_threshold,
    allow_camera_rescale=False,
):
    image_folder = os.path.join(dense_folder, "images_test")
    use_confidence = conf_choose not in [None, "", "none", "None", "null", "NULL"]
    confidence_folder = os.path.join(dense_folder,conf_choose) if use_confidence else None
    cam_folder = os.path.join(dense_folder, "cams_test")
    depth_folder = os.path.join(dense_folder, depth_folder)

    gip_cam = os.path.join(gipuma_folder, "cams")
    gip_img = os.path.join(gipuma_folder, "images")

    os.makedirs(gip_cam, exist_ok=True)
    os.makedirs(gip_img, exist_ok=True)

    images = sorted(f for f in os.listdir(image_folder) if f.endswith(".jpg"))
    prefix_tag = "2333__"


    # depth
    for img in images:
        name = img.replace(".jpg", "")
        sub = os.path.join(gipuma_folder, prefix_tag + name)
        os.makedirs(sub, exist_ok=True)
        im = cv2.imread(os.path.join(image_folder, img))
        if im is None:
            raise FileNotFoundError(os.path.join(image_folder, img))

        depth = np.load(os.path.join(depth_folder, name + ".npy")).astype(np.float32)
        if use_confidence:
            confidence = np.load(os.path.join(confidence_folder, name + ".npy"))
            if confidence.dtype == np.uint8:
                confidence = confidence / 255.0
            if confidence.shape != depth.shape:
                raise ValueError(
                    f"Confidence/depth mismatch for {name}: "
                    f"{confidence.shape} versus {depth.shape}"
                )
            mask = confidence > prob_threshold
            depth[~mask] = 0
        else:
            mask = depth > 0

        H_old, W_old = im.shape[:2]
        H_new, W_new = depth.shape[:2]
        if (H_old, W_old) != (H_new, W_new) and not allow_camera_rescale:
            raise ValueError(
                "Image/depth pixel-grid mismatch for "
                f"{name}: image={(H_old, W_old)}, "
                f"depth={(H_new, W_new)}. MR-MVS does not refusion-rescale "
                "cameras by default."
            )
        if (H_old, W_old) != (H_new, W_new):
            im = cv2.resize(
                im,
                (W_new, H_new),
                interpolation=cv2.INTER_LINEAR,
            )
        cv2.imwrite(os.path.join(gip_img, name + ".png"), im)

        mvsnet_to_gipuma_cam(
            os.path.join(cam_folder, name + "_cam.txt"),
            os.path.join(gip_cam, name + ".png.P"),
            (H_old,W_old),
            (H_new,W_new)
        )

        disp_dmb = os.path.join(sub, "disp.dmb")
        normal_dmb = os.path.join(sub, "normals.dmb")

        write_gipuma_dmb(disp_dmb, depth)
        fake_gipuma_normal(disp_dmb, normal_dmb)


# =========================================================
# Run fusibile
# =========================================================

def run_fusibile(
    gipuma_folder,
    fusibile_exe,
    disp_thresh,
    num_consistent,
    color=True
):
    cam_folder = os.path.join(gipuma_folder, "cams")
    img_folder = os.path.join(gipuma_folder, "images")

    command = [
        fusibile_exe,
        "-input_folder",
        f"{gipuma_folder}/",
        "-p_folder",
        f"{cam_folder}/",
        "-images_folder",
        f"{img_folder}/",
        "--depth_min=0.001",
        "--depth_max=100000",
        "--normal_thresh=360",
        f"--disp_thresh={disp_thresh}",
        f"--num_consistent={num_consistent}",
    ]
    if color:
        command.append("-color_processing")

    print(shlex.join(str(value) for value in command))
    subprocess.run(command, check=True)


# =========================================================
# Top-level API
# =========================================================

def fusion_gipuma(
    dense_folder,
    fusibile_exe,
    depth_folder,
    conf_choose,
    prob_threshold,
    disp_thresh=0.2,
    num_consistent=3,
    allow_camera_rescale=False,
):
    gipuma_folder = os.path.join(dense_folder, "points_gipuma",depth_folder)

    os.makedirs(gipuma_folder, exist_ok=True)

    confidence_mode = (
        conf_choose
        if conf_choose not in (None, "", "none", "None", "null", "NULL")
        else "disabled"
    )
    print(
        "[1] Convert MVSNet depth -> Gipuma "
        f"(confidence: {confidence_mode})"
    )
    mvs_depth_to_gipuma(
        dense_folder,
        gipuma_folder,
        depth_folder,
        conf_choose,
        prob_threshold,
        allow_camera_rescale=allow_camera_rescale,
    )

    print("[2] Run fusibile fusion")
    run_fusibile(
        gipuma_folder,
        fusibile_exe,
        disp_thresh,
        num_consistent
    )

    print("[OK] Fusion finished")


def filter_depth(dense_folder, depth_folder, conf_choose, prob_threshold,
                 n_src_views=10, img_dist_thresh=1.0, depth_thresh=0.01,
                 num_consistent=3, allow_camera_rescale=False):
    plyfilename = os.path.join(dense_folder, depth_folder.split("/")[-1] + ".ply")
    tt_dataset = TTDataset(
        dense_folder,
        depth_folder,
        conf_choose,
        n_src_views=n_src_views,
        allow_camera_rescale=allow_camera_rescale,
    )
    sampler = SequentialSampler(tt_dataset)
    tt_dataloader = DataLoader(
        tt_dataset,
        batch_size=1,
        shuffle=False,
        sampler=sampler,
        num_workers=2,
        pin_memory=True,
        drop_last=False
    )
    views = {}
    for batch_idx, sample_np in enumerate(tt_dataloader):
        sample = tocuda(sample_np)
        for ids in range(sample["src_depths"].size(1)):
            src_prob_mask = sample["src_confs"][:, ids] > prob_threshold
            sample["src_depths"][:, ids, ...] *= src_prob_mask.float()

        prob_mask = sample["ref_conf"] > prob_threshold
        reproj_xyd, in_range = fusion.get_reproj(
            *[sample[attr] for attr in ["ref_depth", "src_depths", "ref_cam", "src_cams"]]
        )
        vis_masks, vis_mask = fusion.vis_filter(
            sample["ref_depth"],
            reproj_xyd,
            in_range,
            img_dist_thresh,
            depth_thresh,
            num_consistent
        )
        ref_depth_ave = fusion.ave_fusion(sample["ref_depth"], reproj_xyd, vis_masks)
        mask = fusion.bin_op_reduce([prob_mask, vis_mask], torch.min)
        idx_img = fusion.get_pixel_grids(*ref_depth_ave.size()[-2:]).unsqueeze(0)
        idx_cam = fusion.idx_img2cam(idx_img, ref_depth_ave, sample["ref_cam"])
        points = fusion.idx_cam2world(idx_cam, sample["ref_cam"])[..., :3, 0].permute(0, 3, 1, 2)

        points_np = points.cpu().data.numpy()
        mask_np = mask.cpu().data.numpy().astype(np.bool_)
        ref_img = sample_np["ref_img"].data.numpy()
        for i in range(points_np.shape[0]):
            p_f = np.stack([points_np[i, k][mask_np[i, 0]] for k in range(3)], -1)
            c_f = np.stack([ref_img[i, k][mask_np[i, 0]] for k in range(3)], -1) * 255
            ref_id = str(sample_np["ref_id"][i].item())
            views[ref_id] = (p_f, c_f.astype(np.uint8))
            print("processing {}, ref-view{:0>2}, photo/geo/final-mask:{}/{}/{}".format(
                dense_folder,
                int(ref_id),
                prob_mask[i].float().mean().item(),
                vis_mask[i].float().mean().item(),
                mask[i].float().mean().item()
            ))

    if len(views) == 0:
        print("[warn] no valid views for pcd fusion:", dense_folder)
        return
    print("Write combined PCD")
    p_all, c_all = [np.concatenate([v[k] for key, v in views.items()], axis=0) for k in range(2)]
    views.clear()
    vertex_all = np.empty(
        len(p_all),
        dtype=[
            ("x", "f4"), ("y", "f4"), ("z", "f4"),
            ("red", "u1"), ("green", "u1"), ("blue", "u1"),
        ],
    )
    for column, prop in enumerate(("x", "y", "z")):
        vertex_all[prop] = p_all[:, column]
    for column, prop in enumerate(("red", "green", "blue")):
        vertex_all[prop] = c_all[:, column]
    el = PlyElement.describe(vertex_all, "vertex")
    PlyData([el]).write(plyfilename)
    print("saving the final model to", plyfilename)


def dynamic_filter_depth(
    dense_folder,
    depth_folder,
    conf_choose,
    prob_threshold,
    n_src_views,
    validated_direct_io=False,
    preload_depths=False,
    preload_fusion_inputs=False,
    dist_base=4.0,
    rel_diff_base=1300.0,
    allow_camera_rescale=False,
):
    plyfilename = os.path.join(dense_folder,depth_folder.split("/")[-1]+'.ply')
    tt_dataset = TTDataset(
        dense_folder,
        depth_folder,
        conf_choose,
        n_src_views=n_src_views,
        load_src_confs=False,
        validated_direct_io=validated_direct_io,
        preload_depths=preload_depths,
        preload_fusion_inputs=preload_fusion_inputs,
        allow_camera_rescale=allow_camera_rescale,
    )
    sampler = SequentialSampler(tt_dataset)
    fast_preloaded = preload_depths and preload_fusion_inputs
    worker_count = 0 if fast_preloaded else 2
    tt_dataloader = DataLoader(
        tt_dataset,
        batch_size=1,
        shuffle=False,
        sampler=sampler,
        num_workers=worker_count,
        pin_memory=not fast_preloaded,
        drop_last=False,
        collate_fn=collate_single_no_copy if fast_preloaded else None,
    )

    ply_writer = OnlineVertexPlyWriter(plyfilename)
    written_views = 0

    for batch_idx, sample_np in enumerate(tt_dataloader):
        num_src_views = sample_np['src_depths'].shape[1]
        dy_range = num_src_views + 1
        sample = tocuda(sample_np)

        ref_depth = sample['ref_depth']  # [1,1,H,W]

        # ----------------------------
        # Reproject source depths into the reference view.
        # ----------------------------
        reproj_xyd = fusion.get_reproj_dynamic(
            *[sample[attr] for attr in
              ['ref_depth', 'src_depths', 'ref_cam', 'src_cams']]
        )

        # ----------------------------
        # Apply dynamic geometric-consistency filtering.
        # ----------------------------
        vis_masks, vis_mask = fusion.vis_filter_dynamic(
            sample['ref_depth'],
            reproj_xyd,
            dist_base=dist_base,
            rel_diff_base=rel_diff_base,
        )

        # ----------------------------
        # Average the geometrically consistent depths.
        # ----------------------------
        if vis_mask is None or vis_mask.numel() == 0 or vis_mask.shape[2] == 0:
            continue
        reproj_depth = reproj_xyd[:, :, -1]  # [1,V,H,W]
        reproj_depth[~vis_mask.squeeze(2)] = 0

        geo_mask_sums = vis_masks.sum(dim=1)
        geo_mask_sum = vis_mask.sum(dim=1)

        depth_est_averaged = (
            torch.sum(reproj_depth, dim=1, keepdim=True) + ref_depth
        ) / (geo_mask_sum + 1)


        geo_mask = geo_mask_sum >= dy_range
        for i in range(2, dy_range):
            geo_mask = torch.logical_or(
                geo_mask,
                geo_mask_sums[:, i - 2] >= i
            )

        # ----------------------------
        # Combine photometric confidence and geometric consistency.
        # ----------------------------
        prob_mask = sample["ref_conf"]>prob_threshold
        mask = fusion.bin_op_reduce([prob_mask, geo_mask], torch.min)
        # mask = geo_mask

        # ----------------------------
        # Back-project retained depths into world coordinates.
        # ----------------------------
        idx_img = fusion.get_pixel_grids(
            *depth_est_averaged.size()[-2:]
        ).unsqueeze(0)

        idx_cam = fusion.idx_img2cam(
            idx_img,
            depth_est_averaged,
            sample['ref_cam']
        )

        points = fusion.idx_cam2world(
            idx_cam,
            sample['ref_cam']
        )[..., :3, 0].permute(0, 3, 1, 2)

        mask_np = mask.cpu().data.numpy().astype(bool)

        ref_img = sample_np['ref_img'].data.numpy()
        _, _, H_d, W_d = sample['ref_depth'].shape
        if tuple(ref_img.shape[-2:]) == (H_d, W_d):
            ref_img_resized = ref_img
        else:
            ref_img_resized = np.zeros(
                (ref_img.shape[0], 3, H_d, W_d), dtype=ref_img.dtype
            )
            for b in range(ref_img.shape[0]):
                for c in range(3):
                    ref_img_resized[b, c] = cv2.resize(
                        ref_img[b, c],
                        (W_d, H_d),
                        interpolation=cv2.INTER_LINEAR
                    )

        for i in range(points.shape[0]):
            # Select on GPU first so only retained points cross to host memory.
            p_f = (
                points[i]
                .permute(1, 2, 0)[mask[i, 0]]
                .detach()
                .cpu()
                .numpy()
            )

            c_f_list = [
                ref_img_resized[i, k][mask_np[i, 0]]
                for k in range(3)
            ]
            c_f = np.stack(c_f_list, -1) * 255

            ref_id = str(sample_np['ref_id'][i].item())
            ply_writer.append(p_f, c_f.astype(np.uint8))
            written_views += 1

            print(
                "processing {}, ref-view{:0>2}, geo-mask:{:.4f}".format(
                    dense_folder,
                    int(ref_id),
                    geo_mask[i].float().mean().item()
                ),
                flush=True,
            )

    # ----------------------------
    # Finalize the streamed PLY.
    # ----------------------------
    print('Write combined PCD', flush=True)

    if written_views == 0:
        ply_writer.abort()
        print("[warn] no valid views for dpcd fusion:", dense_folder)
        return

    vertex_count = ply_writer.close()

    print(
        "saving the final model to",
        plyfilename,
        f"vertices={vertex_count}",
        flush=True,
    )


# =========================================================
# CLI
# =========================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Fuse MR-MVS NPY depth maps with Gipuma/Fusibile"
    )

    parser.add_argument(
        "--dense_folder",
        type=str,
        required=True,
        help="Scene output folder containing depth_est/images_test/cams_test"
    )

    parser.add_argument(
        "--fusibile_exe",
        type=str,
        default="fusibile",
        help="Path to the fusibile executable"
    )

    parser.add_argument(
        "--disp_thresh",
        type=float,
        default=0.1,
        help="Depth/disparity consistency threshold (default: 0.1)"
    )

    parser.add_argument(
        "--num_consistent",
        type=int,
        default=2,
        help="Minimum number of consistent views (default: 2)"
    )

    parser.add_argument(
        "--depth_folder",
        type=str,
        default="depth_est",
        help="Depth-map directory inside dense_folder"
    )

    parser.add_argument(
        "--filter_method",
        type=str,
        default="gipuma",
        choices=("gipuma", "pcd", "dpcd"),
        help="Fusion implementation (default: gipuma)"
    )

    parser.add_argument(
    "--dist_base",
    type=float,
    default=4.0,
    help="DPCD absolute distance threshold"
    )

    parser.add_argument(
        "--rel_diff_base",
        type=float,
        default=1300,
        help="DPCD relative depth difference threshold"
    )

    parser.add_argument(
        "--conf_choose",
        type=str,
        default="confidence",
    )

    parser.add_argument(
        "--prob_threshold",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--fusion_views",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--depth_thresh",
        type=float,
        default=0.01,
    )
    parser.add_argument(
        "--validated_direct_io",
        action="store_true",
        help="Trust validated images_test/cams_test/depth maps to share one pixel grid.",
    )
    parser.add_argument(
        "--preload_depths",
        action="store_true",
        help="Read each depth map once into a fork-shared, read-only fusion cache.",
    )
    parser.add_argument(
        "--preload_fusion_inputs",
        action="store_true",
        help="Preload validated cameras plus reference JPEG bytes and confidence maps.",
    )
    parser.add_argument(
        "--allow_camera_rescale",
        action="store_true",
        help=(
            "Compatibility mode for old outputs whose image and depth sizes "
            "differ. New MR-MVS outputs should never need this."
        ),
    )

    args = parser.parse_args()

    if args.filter_method == "gipuma":
        fusion_gipuma(
            dense_folder=args.dense_folder,
            fusibile_exe=args.fusibile_exe,
            disp_thresh=args.disp_thresh,
            num_consistent=args.num_consistent,
            depth_folder = args.depth_folder,
            conf_choose = args.conf_choose,
            prob_threshold = args.prob_threshold,
            allow_camera_rescale=args.allow_camera_rescale,
        )
    elif args.filter_method == "pcd":
        filter_depth(
            dense_folder=args.dense_folder,
            depth_folder=args.depth_folder,
            conf_choose=args.conf_choose,
            prob_threshold=args.prob_threshold,
            n_src_views=args.fusion_views,
            img_dist_thresh=args.disp_thresh,
            depth_thresh=args.depth_thresh,
            num_consistent=args.num_consistent,
            allow_camera_rescale=args.allow_camera_rescale,
        )
    elif args.filter_method == "dpcd":
        dynamic_filter_depth(
            dense_folder=args.dense_folder,
            depth_folder=args.depth_folder,
            conf_choose = args.conf_choose,
            prob_threshold = args.prob_threshold,
            n_src_views=args.fusion_views,
            validated_direct_io=args.validated_direct_io,
            preload_depths=args.preload_depths,
            preload_fusion_inputs=args.preload_fusion_inputs,
            dist_base=args.dist_base,
            rel_diff_base=args.rel_diff_base,
            allow_camera_rescale=args.allow_camera_rescale,
        )
