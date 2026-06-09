#!/bin/bash
# Image2LoRA 推理启动脚本
set -e
source "$(dirname "$0")/env.sh"

cd "${IMAGE2LORA_ROOT}"

run_python scripts/infer.py \
    --checkpoint_dir outputs/image2lora_sd15/checkpoint-1000 \
    --ref_image examples/ref.jpg \
    --prompt "a beautiful landscape painting in the style of the reference" \
    --output outputs/result.png \
    "$@"
