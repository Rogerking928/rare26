# RARE26 — Frozen foundation-model features with a linear probe

Team **rogerking**, RARE26 Challenge (MICCAI / EndoVis 2026):
binary classification of early neoplasia versus non-dysplastic Barrett's oesophagus
in endoscopic images.

Everything here is released under the **MIT licence** (see `LICENSE`).

## Method in one paragraph

Two frozen ImageNet/LVD-pretrained backbones — DINOv2 ViT-B/14 at 224 px and
ConvNeXtV2-Tiny — produce 768-dimensional pooled features under horizontal-flip
test-time augmentation. Features are L2-normalised, standardised, and scored by a ridge
classifier (`alpha = 3333.3`, `class_weight = None`). Each member's decision function is
standardised by its **training-set** mean and standard deviation, the two z-scores are
averaged, and a sigmoid produces the output probability.

Two design choices are driven by the evaluation setup rather than by accuracy:

- **Anisotropic squash to 224×224**, not resize-and-centre-crop. A gradient-anisotropy
  comparison indicated the organisers rescale test images anisotropically to 512×512;
  cropping would produce a geometry the model never sees at test time.
- **z-standardisation, not rank averaging, for fusion.** The inference container receives
  one case (16 images) at a time, so a within-batch rank is not the rank within the full
  test set. Training-set z-scores are computable per sample and independent of batch size.

The evaluation metric is PPV at 90 % recall at a resampled prevalence of 1 %, which reduces
to `PPV ≈ 0.009 / (0.009 + 0.99·FPR)`. The task is therefore false-positive-rate
minimisation at a fixed operating point; model selection used FPR@90R, never AUROC.

## Repository layout

```
code/scorer.py                      official metric, reimplemented line by line, plus paired bootstrap
code/analysis/extract_feats.py      frozen-backbone feature extraction (--transform squash is required)
code/analysis/extract_dino_pool.py  DINOv2 CLS and mean-pool features from one forward pass
code/analysis/extract_gastronet.py  GastroNet DINOv2 checkpoint loading (domain-pretrained weights)
code/analysis/evaluate2.py          leave-one-centre-out evaluation, per-centre scoring, minimax FPR
code/analysis/experiments.py        the pre-specified candidate experiments and acceptance gate
code/analysis/gastronet_gate.py     pre-committed decision rule for domain-pretrained weights
code/analysis/export_ensemble.py    fits the final model and packs it into the container resources
code/submission/                    the algorithm container (inference.py, model/, Dockerfile)
```

`code/analysis/probe.py`, `compare.py`, `heads.py` and `ensemble.py` are earlier exploratory
scripts. **Their numbers are superseded**: they pooled centres after a within-centre rank
transform, which is not reproducible at test time and inflated estimates by roughly 30 %.
They are retained for provenance. Use `evaluate2.py`.

## Reproducing

Data access requires registration and a data use agreement with the challenge organisers;
the images are not redistributed here. Place them as `data/center_{1,2}/{ndbe,neo}/*.png`.

```bash
pip install -r code/submission/requirements.txt

# 1. extract frozen features (CPU, ~16 min per backbone for 3,095 images)
python3 code/analysis/extract_feats.py --model vit_base_patch14_dinov2.lvd142m --img-size 224 --hflip
python3 code/analysis/extract_feats.py --model convnextv2_tiny.fcmae_ft_in22k_in1k --hflip

# 2. leave-one-centre-out evaluation
python3 code/analysis/evaluate2.py

# 3. candidate experiments against the pre-specified gate
python3 code/analysis/experiments.py

# 4. fit the final model and pack it into the container
python3 code/analysis/export_ensemble.py --members dino224 convnextv2
```

### Container

```bash
cd code/submission
docker build --platform=linux/amd64 -t rare26 .

IN="$PWD/test/input/interface_0"; OUT="$PWD/test/output/interface_0"
docker run --rm --platform=linux/amd64 --network none \
  --volume "$IN":/input:ro --volume "$OUT":/output rare26
```

Input is a stacked TIFF or MHA at
`/input/images/stacked-barretts-esophagus-endoscopy/`; output is
`/output/stacked-neoplastic-lesion-likelihoods.json`, one probability per image.

Measured on two pinned vCPUs, end-to-end per case of 16 images: median 29.0 s, 95th
percentile 35.8 s, peak resident memory 1.67 GB.

## Licensing note

`code/submission/inference.py` and its `Dockerfile` are **original implementations**. The
organisers' submission template is distributed under CC BY-NC 4.0, which is incompatible
with the MIT licence the challenge requires for leaderboard eligibility, so no template code
is reused here. The organisers' baseline and template repositories are not redistributed;
obtain them from the challenge organisers.

Model weights are the public timm checkpoints named above and carry their own licences.

## Environment

Python 3.12 for analysis, 3.11 in the container. PyTorch 2.13.0 (CPU build),
torchvision 0.28.0, timm 1.0.28, scikit-learn 1.8.0, NumPy 2.4.6, SimpleITK 2.5.6,
Pillow 12.2.0. No GPU is required or used. Random seeds are fixed at 0.
