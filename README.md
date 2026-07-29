# MR-MVS: Monocular Guidance with Multi-view Geometry Optimization for Multi-view Stereo

### [Project Page](https://zhouannan.github.io/MR-MVS/)

MR-MVS is the minimal runnable implementation used in our experiments. It is
built on MVSFormer++ and contains only:

1. AlphaNet monocular-prior correction at stages 1 to 3.
2. Final-stage multi-view geometric refinement.
3. NPY depth export and Gipuma/Fusibile fusion.

The repository does not depend on SAM, Depth Anything, RoMa, DINOv3, or
sparse COLMAP points.

## Project layout

```text
MR-MVS/
  config/
    mr_mvs_blended.json
    mr_mvs_tnt.json
  datasets/
  models/
    alpha_fusion.py
    cost_volume.py
    geometry.py
    stage_refine.py
    multi_stage_infer.py
    networks/DINOv2_mvsformer_model.py
  tools/merge_checkpoints.py
  train.py
  test.py
  test_stage_refine.py
  fusion.py
```

## Installation

Python 3.10 and a recent CUDA-enabled PyTorch are recommended.

```bash
conda create -n mr-mvs python=3.10 -y
conda activate mr-mvs
pip install -r requirements.txt
```

Optional CUDA attention kernels are listed in
`requirements-optional.txt`; the default implementation does not require
them.

Download the DINOv2 ViT-B/14 checkpoint and set `vit_path` in the JSON config,
or pass `--vit_path` to the test scripts.

## Evaluation data

Each scene follows the MVSNet layout:

```text
TNT_ROOT/
  Auditorium/
    images/
      00000000.jpg
    cams_1/               # cameras used by the released T&T setting
      00000000_cam.txt
    cams/                 # supported alternative layout
    new_pair.txt          # pair.txt is also accepted
MONO_ROOT/
  Auditorium/
    mono_depth/
      00000000.npy
```

`pair.txt` is read from the scene directory. Camera extrinsics are
world-to-camera matrices. The loader applies the Tanks-and-Temples vertical
padding and principal-point update before resizing. Images, monocular depth,
stage-4 cameras, exported depth, and fusion input therefore share one pixel
grid.

For Tanks and Temples, `cams_1/` is tried first to reproduce the released
experiments. Set `MVSFORMER_TT_PREFER_CAMS=1` only for datasets whose corrected
cameras are stored in `cams/`.

## Checkpoints

MR-MVS can load either:

- one merged checkpoint containing MVSFormer++ and AlphaNet; or
- an MVSFormer++ checkpoint plus `--alpha_checkpoint`.

Create the preferred single checkpoint with:

```bash
python tools/merge_checkpoints.py \
  --mvs_checkpoint pretrained_models/tnt/model_best.pth \
  --alpha_checkpoint checkpoints/best_1.pth \
  --output checkpoints/mr_mvs_tnt.pth
```

Stages 1 to 3 must contain trained AlphaNet weights when monocular depth is
enabled. Stage 4 is intentionally not corrected by AlphaNet.

## Training

Place monocular depths in each BlendedMVS scene's `mono_depth/` directory, or
set `mono_depths_path` in `config/mr_mvs_blended.json`.

```bash
python train.py \
  --config config/mr_mvs_blended.json \
  --pretrained pretrained_models/mvsformerpp_blended.pth \
  --data_path /path/to/BlendedMVS \
  --mono_depths_path /path/to/mono_depths \
  --exp_name mr_mvs
```

For distributed training:

```bash
python train.py --config config/mr_mvs_blended.json --ddp ...
```

The saved `model_best.pth` and `model_last.pth` already include AlphaNet, so
later inference needs only one checkpoint.

## Depth inference

MVSFormer++ baseline, without monocular depth:

```bash
python test.py \
  --config config/mr_mvs_tnt.json \
  --model checkpoints/mr_mvs_tnt.pth \
  --testpath /path/to/TanksAndTemples \
  --testlist lists/tanksandtemples/advanced.txt \
  --outdir outputs/tnt \
  --dataset tt --max_h 1088 --max_w 1920 --num_view 20
```

AlphaNet-guided MR-MVS, without geometric refinement:

```bash
python test.py \
  --config config/mr_mvs_tnt.json \
  --model checkpoints/mr_mvs_tnt.pth \
  --testpath /path/to/TanksAndTemples \
  --testlist lists/tanksandtemples/advanced.txt \
  --mono_depths_path /path/to/mono_depths \
  --outdir outputs/tnt \
  --dataset tt --max_h 1088 --max_w 1920 --num_view 20
```

## AlphaNet plus geometric refinement

The experiment setting is 10 iterations and a 0.01-pixel reprojection
threshold. Fifty reference views are processed per disk-output chunk:

```bash
python test_stage_refine.py \
  --config config/mr_mvs_tnt.json \
  --model checkpoints/mr_mvs_tnt.pth \
  --testpath /path/to/TanksAndTemples \
  --testlist lists/tanksandtemples/advanced.txt \
  --mono_depths_path /path/to/mono_depths \
  --outdir outputs/tnt \
  --dataset tt --max_h 1088 --max_w 1920 --num_view 20 \
  --chunk_size 50 --refine_iters 10 --reproj_threshold 0.01
```

The threshold is applied to every source view independently. A source whose
reprojection error is greater than 0.01 pixel receives zero weight. Errors
from the remaining sources are averaged with learned visibility weights. One
passing source is sufficient.

The released Tanks-and-Temples setting uses a 1088x1920 pixel grid and
`--num_view 20` (one reference plus 19 source views). The same resized images
and camera matrices written by inference are consumed by fusion.

Both test scripts write:

```text
outputs/tnt/Auditorium/
  depth_est/*.npy
  confidence/*.npy
  stage_4_confidence/*.npy
  images/ and images_test/
  cams/ and cams_test/
```

## Fusion

The default command matches the commonly used Gipuma setting
`disp_thresh=0.1`, `num_consistent=2`:

```bash
python fusion.py \
  --dense_folder outputs/tnt/Auditorium \
  --fusibile_exe /path/to/fusibile \
  --filter_method gipuma \
  --depth_folder depth_est \
  --conf_choose confidence \
  --prob_threshold 0.5 \
  --disp_thresh 0.1 \
  --num_consistent 2
```

Fusion rejects image/depth size mismatches by default. This prevents an
implicit "refusion" camera rescale. `--allow_camera_rescale` exists only for
compatibility with old output directories.

## AlphaNet details

For every corrected stage, AlphaNet consumes eight per-pixel channels:
confidence, entropy, top-2 margin, log MVS depth, log aligned monocular depth,
signed residual, absolute residual, and the prior mask. It has two 3x3
convolution layers with 32 channels and separate 1x1 heads for alpha and
sigma.

Monocular depth is first scale-shift aligned to MVS depth and then affinely
aligned in 3D using MVS pixels above the 0.5 confidence threshold. A
high-confidence pixel is considered an outlier when its residual differs
from the reliable-set mean by more than one standard deviation. The prior
mask is the union of low-confidence pixels and these residual outliers.

The Gaussian correction is added to cost-volume logits, not to normalized
probabilities. A softmax is then applied along the depth dimension, so the
final probability volume is normalized.

`--alpha_scale` changes the learned correction strength at inference. The
default is 1.0; for example, `--alpha_scale 0.01` reproduces the reduced-mono
influence experiment.

## Tests

```bash
python -m unittest discover -s tests -v
```

The tests cover probability normalization, AlphaNet gradients, checkpoint
key handling, per-source reprojection gating, and planar-depth stability.

## Acknowledgements

MR-MVS is built on
[MVSFormer++](https://github.com/maybeLx/MVSFormerPlusPlus) and uses components
from [DINOv2](https://github.com/facebookresearch/dinov2). We thank the
authors for releasing their code and models.

## License

This project is released under the Apache License 2.0. See [LICENSE](LICENSE).
