# Evaluation README

This repository evaluates generated style-transfer images with `scripts/evaluate.py`.
The script does not download checkpoints or metric weights. Prepare all required
packages and checkpoints before running it.

## Metric Support

| Metric | Content image dependency | Computed by `scripts/evaluate.py` | Implementation |
|---|---|---:|---|
| `Style Loss` | Not needed | Yes | `style_loss_gram` uses the original Gatys VGG Gram formula; `style_loss_adain` is an extra AdaIN mean/std statistic |
| `FID` | Not needed | Yes | Calls `pytorch-fid` |
| `SSIM` | Required | Yes, only when content images are supplied | Calls `skimage.metrics.structural_similarity` |
| `LPIPS` | Required for content fidelity | Yes, only when content images are supplied | Calls `lpips` |
| `ArtFID` | Required | Yes, only when content images are supplied | Calls `art-fid` |

In the current no-content setting, the meaningful metrics are `Style Loss` and
`FID`. `SSIM`, `LPIPS`, and `ArtFID` are skipped unless a `content` path is
provided in the manifest or a content image is present in each case directory.

## Package Setup

Install the metric packages in the same Python environment used to run
evaluation:

```bash
pip install torch torchvision pillow numpy
pip install pytorch-fid
pip install scikit-image lpips art-fid
```

`pytorch-fid`, `lpips`, and `art-fid` are external metric implementations. The
evaluation script calls them instead of reimplementing those metrics.

## Checkpoints

### VGG19 for Style Loss

Required for:

- `style_loss_gram`
- `style_loss_adain`

Checkpoint:

- Version: `torchvision.models.VGG19_Weights.IMAGENET1K_V1`
- Publisher: PyTorch / torchvision
- Training data: ImageNet-1K
- Filename: `vgg19-dcbb9e9d.pth`
- Official URL: <https://download.pytorch.org/models/vgg19-dcbb9e9d.pth>
- Hugging Face model card using torchvision weights: <https://huggingface.co/timm/vgg19.tv_in1k>

Recommended local placement:

```bash
mkdir -p pretrained_models/metrics
curl -L \
  -o pretrained_models/metrics/vgg19-dcbb9e9d.pth \
  https://download.pytorch.org/models/vgg19-dcbb9e9d.pth
```

Then pass the file explicitly:

```bash
python scripts/evaluate.py \
  --cases_dir image_in_ppt \
  --output_dir outputs/eval \
  --style_loss_vgg_weights pretrained_models/metrics/vgg19-dcbb9e9d.pth
```

If `--style_loss_vgg_weights` is omitted, the script looks for the standard
torch cache file:

```text
${TORCH_HOME:-~/.cache/torch}/hub/checkpoints/vgg19-dcbb9e9d.pth
```

### Inception for FID

Required for:

- `FID`

Used by:

- `pytorch-fid`

Checkpoint:

- Version: `pt_inception-2015-12-05-6726825d`
- Publisher: `mseitzer/pytorch-fid`, converted from the TensorFlow Inception checkpoint used for FID
- Filename: `pt_inception-2015-12-05-6726825d.pth`
- Official URL: <https://github.com/mseitzer/pytorch-fid/releases/download/fid_weights/pt_inception-2015-12-05-6726825d.pth>
- Hugging Face address: none found for the canonical `pytorch-fid` checkpoint

Recommended local placement for `pytorch-fid`:

```bash
mkdir -p "${TORCH_HOME:-$HOME/.cache/torch}/hub/checkpoints"
curl -L \
  -o "${TORCH_HOME:-$HOME/.cache/torch}/hub/checkpoints/pt_inception-2015-12-05-6726825d.pth" \
  https://github.com/mseitzer/pytorch-fid/releases/download/fid_weights/pt_inception-2015-12-05-6726825d.pth
```

Alternatively, place it anywhere and pass it explicitly:

```bash
python scripts/evaluate.py \
  --cases_dir image_in_ppt \
  --output_dir outputs/eval \
  --fid_weights pretrained_models/metrics/pt_inception-2015-12-05-6726825d.pth
```

The evaluation script checks for local FID weights before calling
`pytorch-fid`, so it will not trigger an implicit GitHub download during metric
calculation.

FID is skipped when either compared image set has fewer than 2 images, because
the covariance estimate becomes degenerate and `pytorch-fid` can fail with NaNs.
Override the threshold with `--fid_min_images` only if you know what you are
doing.

FID defaults to `--fid_batch_size 1` so mixed-resolution images do not fail
during PyTorch collation. Increase it only when all images in both compared
sets have the same dimensions.

### ArtFID Checkpoint

Required for:

- `ArtFID`

Used by:

- `art-fid`

Checkpoint:

- Version: `art_inception.pth`
- Publisher: Matthias Wright / ArtFID
- Filename: `art_inception.pth`
- Hugging Face model: <https://huggingface.co/matthias-wright/art_inception>
- Checkpoint URL: <https://huggingface.co/matthias-wright/art_inception/resolve/main/art_inception.pth>

`ArtFID` also needs content images. Without content images, `scripts/evaluate.py`
marks it as skipped.

### LPIPS Weights

Required for:

- `LPIPS`
- ArtFID content metric when `art-fid` uses LPIPS

Checkpoint/package details:

- Version: LPIPS `v0.1`
- Publisher: Richard Zhang et al. / `richzhang/PerceptualSimilarity`
- Package repository: <https://github.com/richzhang/PerceptualSimilarity>
- Filenames inside the package: `weights/v0.1/alex.pth`, `weights/v0.1/vgg.pth`, `weights/v0.1/squeeze.pth`
- Hugging Face address: none found for the canonical LPIPS package weights

The learned LPIPS linear weights are distributed with the `lpips` package. If
your package version or backbone uses torchvision trunk weights, prepare those
weights in the torch cache before running evaluation.

## Directory-Based Evaluation

The current repository already has directories like:

```text
image_in_ppt/<prompt>/output.png
image_in_ppt/<prompt>/baseline.png
image_in_ppt/<prompt>/<style-reference>.jpg
```

Run no-content evaluation:

```bash
python scripts/evaluate.py \
  --cases_dir image_in_ppt \
  --output_dir outputs/eval \
  --style_loss_vgg_weights pretrained_models/metrics/vgg19-dcbb9e9d.pth
```

To skip FID and only compute style loss:

```bash
python scripts/evaluate.py \
  --cases_dir image_in_ppt \
  --output_dir outputs/eval_style_loss \
  --skip_fid \
  --skip_content_metrics \
  --style_loss_vgg_weights pretrained_models/metrics/vgg19-dcbb9e9d.pth
```

Outputs:

```text
outputs/eval/metrics_report.json
outputs/eval/style_loss.csv
```

If FID runs successfully, its values are stored in `metrics_report.json`.

## Manifest-Based Evaluation

Use a JSONL manifest when outputs are not arranged as one directory per prompt.

No-content record:

```json
{"prompt": "a river in the mountain", "style_ref": "path/to/style.jpg", "generated": "path/to/output.png", "method": "image2lora"}
```

Record with content image:

```json
{"prompt": "a river in the mountain", "style_ref": "path/to/style.jpg", "content": "path/to/content.jpg", "generated": "path/to/output.png", "method": "image2lora"}
```

Run:

```bash
python scripts/evaluate.py \
  --manifest outputs/eval_manifest.jsonl \
  --output_dir outputs/eval \
  --style_loss_vgg_weights pretrained_models/metrics/vgg19-dcbb9e9d.pth
```

## Notes

- Lower is better for `style_loss_gram`, `style_loss_adain`, `FID`, `LPIPS`, and `ArtFID`.
- Higher is better for `SSIM`.
- `style_loss_gram` follows the original Gatys formula: `sum_l w_l * E_l`, `E_l = 1 / (4 * N_l^2 * M_l^2) * sum_ij((G_l - A_l)^2)`, with VGG `conv1_1` through `conv5_1`, equal layer weight `w_l = 1/5`, and average pooling on the Gatys feature path.
- FID is distribution-level and is weak with very small image sets. Use enough generated images and style references when possible.
- `Style Loss` is computed per generated image against its paired style reference, so it works in the current no-content setting.
