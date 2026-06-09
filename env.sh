#!/bin/bash
# Image2LoRA 环境变量与 Python 包装器（兼容旧版 glibc 系统）
export IMAGE2LORA_ENV="image2lora"
export IMAGE2LORA_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export IMAGE2LORA_PYTHON="/scratch/jiaqi/anaconda3/envs/${IMAGE2LORA_ENV}/bin/python"

# HuggingFace 镜像（预下载模型时生效，可按需取消注释）
# export HF_ENDPOINT="https://hf-mirror.com"

run_python() {
    /scratch/jiaqi/glibc-2.34/lib/ld-linux-x86-64.so.2 \
        --library-path /scratch/jiaqi/glibc-2.34/lib:/scratch/jiaqi/anaconda3/envs/${IMAGE2LORA_ENV}/lib:/lib64 \
        "${IMAGE2LORA_PYTHON}" "$@"
}

export -f run_python
