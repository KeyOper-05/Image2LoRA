# Image2LoRA

基于 Video2LoRA 思路实现的图像风格迁移框架：通过 **Hypernetwork** 从参考图像（DINOv2 特征）动态生成 **LightLoRA** 权重，注入冻结的 **Stable Diffusion 1.5**，实现高层语义风格控制。

## 架构

```
参考图 I_ref ──► DINOv2 Encoder ──► Hypernetwork (Transformer Decoder + 迭代细化)
                                           │
                                           ▼
                                    LightLoRA 权重 (~50KB)
                                           │
目标描述 + 噪声 ──► SD 1.5 UNet (+ LightLoRA) ──► 风格化图像
```

## 环境配置

```bash
# 已创建 conda 环境 image2lora
source env.sh

# 使用包装后的 Python（兼容旧 glibc）
run_python -c "import torch; print(torch.__version__)"
```

所有 Python 命令请通过 `source env.sh && run_python ...` 调用。

## 预下载 SD 1.5（HuggingFace 镜像）

diffusers 的 `from_pretrained()` 支持**本地目录**，下载一次后可离线加载。

### 方式一：一键脚本（推荐）

```bash
source env.sh

# 使用 hf-mirror 镜像（可在 env.sh 中设置 HF_ENDPOINT）
bash scripts/download_sd15.sh
```

模型会保存到 `pretrained_models/stable-diffusion-v1-5/`，目录结构如下：

```
pretrained_models/stable-diffusion-v1-5/
├── tokenizer/
├── text_encoder/
├── vae/
├── unet/
└── scheduler/
```

### 方式二：手动下载

```bash
source env.sh
export HF_ENDPOINT="https://hf-mirror.com"

run_python -c "
from huggingface_hub import snapshot_download
snapshot_download('runwayml/stable-diffusion-v1-5',
    local_dir='pretrained_models/stable-diffusion-v1-5',
    local_dir_use_symlinks=False)
"
```

### 预下载 DINOv2（HuggingFace 镜像）

`dinov2_vitb14` 对应 `facebook/dinov2-base`，约 **350 MB**。

```bash
source env.sh
export HF_ENDPOINT="https://hf-mirror.com"
bash scripts/download_dinov2.sh
```

或手动下载：

```bash
source env.sh
export HF_ENDPOINT="https://hf-mirror.com"
run_python -c "
from huggingface_hub import snapshot_download
snapshot_download('facebook/dinov2-base',
    local_dir='pretrained_models/dinov2-base',
    allow_patterns=['config.json','model.safetensors','preprocessor_config.json'])
"
```

配置 `configs/train_sd15.yaml`：

```yaml
dinov2_model_path: "pretrained_models/dinov2-base"
```

训练/推理时会自动 `local_files_only=True` 从本地加载，不再访问网络。

### 加载本地模型

修改 `configs/train_sd15.yaml`：

```yaml
pretrained_model_name_or_path: "pretrained_models/stable-diffusion-v1-5"
```

或通过命令行覆盖（无需改配置文件）：

```bash
# 训练
run_python scripts/train.py \
    --pretrained_model_name_or_path pretrained_models/stable-diffusion-v1-5

# 推理
run_python scripts/infer.py \
    --pretrained_model pretrained_models/stable-diffusion-v1-5 \
    --checkpoint_dir pretrained_models/ckpt \
    --ref_image examples/ref.jpg \
    --prompt "a landscape painting" \
    --output outputs/result.png
```

> 路径可以是相对路径（相对项目根目录）或绝对路径，例如 `/scratch/jiaqi/image2lora/pretrained_models/stable-diffusion-v1-5`。

## 数据准备（10k-15k 对）

metadata.jsonl 每行格式：

```json
{"ref_image": "styles/watercolor_01.jpg", "tgt_image": "targets/watercolor_01.jpg", "caption": "a cat in watercolor style", "class": "watercolor"}
```

### 方式一：content/styled 目录对

```bash
run_python scripts/prepare_dataset.py \
    --content_dir /path/to/content \
    --styled_dir /path/to/styled \
    --data_root data \
    --output data/metadata.jsonl \
    --max_samples 15000 \
    --make_relative
```

### 方式二：StyleBooth 目录结构

```bash
run_python scripts/prepare_dataset.py \
    --stylebooth_dir /path/to/stylebooth \
    --data_root data \
    --output data/metadata.jsonl \
    --max_samples 15000 \
    --make_relative
```

推荐数据集：OmniStyle-1M、StyleBooth、SPair-71k（采样 10k-15k 对，覆盖 200+ 类别）。

## 训练

```bash
source env.sh
bash train.sh
```

或自定义参数：

```bash
run_python scripts/train.py \
    --config configs/train_sd15.yaml \
    --train_batch_size 2 \
    --learning_rate 1e-4
```

训练配置见 `configs/train_sd15.yaml`。默认：
- 基座：`runwayml/stable-diffusion-v1-5`
- LightLoRA：rank=1, down_dim=64, up_dim=32
- Hypernetwork：8 层 decoder, 4 次迭代细化
- 损失：标准扩散噪声预测 MSE（与 Video2LoRA 一致）

Checkpoint 保存在 `outputs/image2lora_sd15/checkpoint-{step}/`，包含：
- `lora_aux.safetensors`：LightLoRA 辅助矩阵
- `hypernetwork.safetensors`：超网络权重

## 推理

```bash
source env.sh
run_python scripts/infer.py \
    --checkpoint_dir pretrained_models/ckpt \
    --ref_image examples/ref.jpg \
    --prompt "a handsome man in the style of the reference" \
    --output outputs/result.png \
    --num_inference_steps 30 \
    --guidance_scale 7.5
```

### 批量推理

`scripts/batch_infer.py` 可以遍历内容图片和风格参考图片，以每个 style 为一组，
为该 style 生成全体 caption 对应的图片，并写出结果 manifest。

默认支持一个目录里同时放内容图和风格参考图：

- 内容图：`类型_caption.ext`，例如 `ancientbuilding_A beautifully preserved Greek Parthenon at twilight.png`
- 风格参考：`style____*.jpg`，例如 `s0055____1019_01_query_0_img_000004_1684018013820_07938481893127348.jpg`
- 生成图：`内容图文件名&&风格参考文件名.jpg`

```bash
source env.sh
run_python scripts/batch_infer.py \
    --input_dir /path/to/images \
    --output_dir outputs/batch_results \
    --checkpoint_dir outputs/image2lora_sd15/checkpoint-1000
```

如果内容图和风格参考图在不同目录：

```bash
source env.sh
run_python scripts/batch_infer.py \
    --content_dir /path/to/content_images \
    --style_dir /path/to/style_refs \
    --output_dir outputs/batch_results \
    --checkpoint_dir outputs/image2lora_sd15/checkpoint-1000
```

默认 prompt 直接使用内容图文件名里的 `caption`。如需追加风格描述，可传模板：

```bash
--prompt_template "{caption} in the style of the reference"
```

manifest 默认写到 `outputs/batch_results/manifest.json`，每条记录包含
`content_image`、`content_type`、`caption`、`prompt`、`style_image`、`style`、
`output_image` 和 `status`。正式生成前可先用 `--dry_run` 检查解析和命名。

### 批量评估

`scripts/batch_eval.py` 使用和批量推理相同的命名规则。默认会评估已经存在的生成图：
某个 style 只要有 1 张生成图就参与评估，0 张生成图的 style 会被忽略。它会按
style 分组计算 Style Loss 和 FID，并写出 `batch_metrics_report.json`。同一
style 前缀下的多张参考图会合并为该 style 的 FID 参考集合。

使用批量推理的 manifest：

```bash
source env.sh
run_python scripts/batch_eval.py \
    --batch_manifest outputs/batch_results/manifest.json \
    --output_dir outputs/batch_eval \
    --fid_style_ref_dir /path/to/fid_style_refs \
    --fid_weights pretrained_models/metrics/pt_inception-2015-12-05-6726825d.pth \
    --style_loss_vgg_weights pretrained_models/metrics/vgg19-dcbb9e9d.pth
```

或按目录和命名规则重新发现生成图：

```bash
source env.sh
run_python scripts/batch_eval.py \
    --input_dir /path/to/images \
    --generated_dir outputs/batch_results \
    --output_dir outputs/batch_eval \
    --fid_style_ref_dir /path/to/fid_style_refs \
    --fid_weights pretrained_models/metrics/pt_inception-2015-12-05-6726825d.pth \
    --style_loss_vgg_weights pretrained_models/metrics/vgg19-dcbb9e9d.pth
```

如果 FID 的参考图在单独目录中，文件名可形如
`ancientbuilding_A beautifully preserved Greek Parthenon at twilight.png&&s0172____1107_01_query_1_img_000018_1683099425957_08968873238001791.jpg.jpg.png`。
脚本会从 `&&` 后提取 style 前缀，并把同一 style 下的这些图片作为 FID reference set。

FID 使用 `pytorch-fid` 的 Inception 权重，文件名为
`pt_inception-2015-12-05-6726825d.pth`。建议提前下载到
`pretrained_models/metrics/` 并通过 `--fid_weights` 传入，避免评估时联网下载。
如果某个 style 的生成图或参考图少于 2 张，FID 会被跳过以避免协方差退化。
FID 默认使用 `--fid_batch_size 1` 以兼容不同分辨率图片；若图片尺寸完全一致，可调大该值提速。
如果只想快速计算 Style Loss，可加 `--skip_fid`。

如果需要严格检查每个 style 都有全体 caption 的生成图：

```bash
--require_complete
```

输出包括：

- `outputs/batch_eval/batch_metrics_report.json`：每个 style 的 FID、Style Loss 摘要
- `outputs/batch_eval/style_loss.csv`：逐张生成图的 Style Loss
- `outputs/batch_eval/eval_manifest.jsonl`：传给指标计算的评估清单
- `outputs/batch_eval/by_style/<style>/metrics_report.json`：单个 style 的完整报告

## 项目结构

```
image2lora/
├── configs/train_sd15.yaml      # 训练配置
├── image2lora/
│   ├── models/
│   │   ├── attention.py       # Transformer decoder
│   │   ├── hypernet.py        # ImageHyperDream
│   │   ├── lora.py            # LightLoRA (SD 1.5 UNet)
│   │   └── encoder.py         # DINOv2 编码器
│   └── data/dataset.py        # 成对图像数据集
├── scripts/
│   ├── prepare_dataset.py     # 数据准备
│   ├── train.py               # 训练
│   └── infer.py               # 推理
├── env.sh                     # Python 包装器
├── train.sh
└── infer.sh
```

## 显存参考

| 配置 | 显存 | 说明 |
|------|------|------|
| batch=2, fp16, 512px | ~18-22 GB | RTX 4090 可训练 |
| batch=1, fp16, 512px | ~14-16 GB | 低显存模式 |

## 与 Video2LoRA 的对应关系

| Video2LoRA | Image2LoRA |
|------------|------------|
| 参考视频 VAE latent | 参考图 DINOv2 特征 |
| VideoHyperDream | ImageHyperDream |
| CogVideoX + LightLoRA | SD 1.5 + LightLoRA |
| 成对视频 (ref, tgt) | 成对图像 (I_ref, I_tgt) |
| 噪声预测 MSE loss | 噪声预测 MSE loss |
