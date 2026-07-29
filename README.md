# MR-MVS: Monocular Guidance with Multi-view Geometry Optimization for Multi-view Stereo

### [Project Page](https://zhouannan.github.io/MR-MVS/)

## Installation

```bash
conda create -n mr-mvs python=3.10 -y
conda activate mr-mvs
pip install -r requirements.txt
```

Download the DINOv2 ViT-B/14 weights and set `vit_path` in the configuration
file or pass it with `--vit_path`.

## Data Preparation

Prepare Tanks and Temples in the following layout:

```text
TNT_ROOT/
  Auditorium/
    images/
      00000000.jpg
    cams_1/
      00000000_cam.txt
    new_pair.txt

MONO_ROOT/
  Auditorium/
    mono_depth/
      00000000.npy
```

Use `pair.txt` instead of `new_pair.txt` when needed. If cameras are stored in
`cams/`, run:

```bash
export MVSFORMER_TT_PREFER_CAMS=1
```

## Training

Single GPU:

```bash
python train.py \
  --config config/mr_mvs_blended.json \
  --pretrained /path/to/mvsformerpp_blended.pth \
  --data_path /path/to/BlendedMVS \
  --mono_depths_path /path/to/mono_depths \
  --exp_name mr_mvs
```

Multiple GPUs:

```bash
CUDA_VISIBLE_DEVICES=0,1 python train.py \
  --config config/mr_mvs_blended.json \
  --pretrained /path/to/mvsformerpp_blended.pth \
  --data_path /path/to/BlendedMVS \
  --mono_depths_path /path/to/mono_depths \
  --exp_name mr_mvs \
  --ddp
```

## Testing

```bash
CUDA_VISIBLE_DEVICES=0 python test_stage_refine.py \
  --config config/mr_mvs_tnt.json \
  --model /path/to/mr_mvs_tnt.pth \
  --vit_path /path/to/dinov2_vitb14_pretrain.pth \
  --testpath /path/to/TanksAndTemples \
  --testlist lists/tanksandtemples/advanced.txt \
  --mono_depths_path /path/to/mono_depths \
  --outdir outputs/tnt \
  --dataset tt \
  --max_h 1088 \
  --max_w 1920 \
  --num_view 20 \
  --numdepth 192 \
  --chunk_size 50 \
  --amp_dtype bf16 \
  --refine_iters 10 \
  --reproj_threshold 0.01
```

Outputs:

```text
outputs/tnt/Auditorium/
  depth_est/
  confidence/
  stage_4_confidence/
  images/
  images_test/
  cams/
  cams_test/
  pair.txt
```

## Fusion

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

## Acknowledgements

MR-MVS is built on
[MVSFormer++](https://github.com/maybeLx/MVSFormerPlusPlus) and uses components
from [DINOv2](https://github.com/facebookresearch/dinov2).

## License

This project is released under the Apache License 2.0. See [LICENSE](LICENSE).
