#!/bin/bash
# Image2LoRA 训练启动脚本
set -e
source "$(dirname "$0")/env.sh"

cd "${IMAGE2LORA_ROOT}"

# 1. 准备数据集（按需修改路径）
# run_python scripts/prepare_dataset.py \
#     --stylebooth_dir /path/to/stylebooth \
#     --data_root data \
#     --output data/metadata.jsonl \
#     --max_samples 15000 \
#     --make_relative

# 2. 开始训练
run_python scripts/train.py \
    --config configs/train_sd15.yaml \
    "$@"
