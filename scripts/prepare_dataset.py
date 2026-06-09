#!/usr/bin/env python3
"""
准备 Image2LoRA 训练数据集 (10k-15k 对)。

支持三种输入格式:
  1. paired_dir: 包含 content/ 和 styled/ 子目录，文件名一一对应
  2. jsonl: 已有 metadata.jsonl
  3. stylebooth: StyleBooth 风格目录结构

输出: data/metadata.jsonl
"""

import argparse
import json
import os
import random
from pathlib import Path


def find_pairs_from_dirs(content_dir: str, styled_dir: str, caption_template: str = ""):
    content_dir = Path(content_dir)
    styled_dir = Path(styled_dir)
    exts = {".jpg", ".jpeg", ".png", ".webp"}

    content_files = {f.stem: f for f in content_dir.iterdir() if f.suffix.lower() in exts}
    pairs = []
    for stem, content_path in sorted(content_files.items()):
        for ext in exts:
            styled_path = styled_dir / f"{stem}{ext}"
            if styled_path.exists():
                pairs.append({
                    "ref_image": str(content_path),
                    "tgt_image": str(styled_path),
                    "caption": caption_template or f"a photo in artistic style, {stem.replace('_', ' ')}",
                    "class": styled_dir.name,
                })
                break
    return pairs


def find_pairs_stylebooth(root_dir: str):
    """StyleBooth 格式: root/<style_name>/content/*.jpg + root/<style_name>/styled/*.jpg"""
    root = Path(root_dir)
    pairs = []
    for style_dir in sorted(root.iterdir()):
        if not style_dir.is_dir():
            continue
        content_dir = style_dir / "content"
        styled_dir = style_dir / "styled"
        if content_dir.exists() and styled_dir.exists():
            pairs.extend(find_pairs_from_dirs(str(content_dir), str(styled_dir)))
    return pairs


def subsample(pairs, max_samples: int, seed: int = 42):
    if max_samples <= 0 or len(pairs) <= max_samples:
        return pairs
    rng = random.Random(seed)
    return rng.sample(pairs, max_samples)


def main():
    parser = argparse.ArgumentParser(description="Prepare Image2LoRA dataset metadata")
    parser.add_argument("--output", type=str, default="data/metadata.jsonl")
    parser.add_argument("--data_root", type=str, default="data")
    parser.add_argument("--content_dir", type=str, default=None, help="原始内容图目录")
    parser.add_argument("--styled_dir", type=str, default=None, help="风格化图目录")
    parser.add_argument("--stylebooth_dir", type=str, default=None, help="StyleBooth 根目录")
    parser.add_argument("--input_jsonl", type=str, default=None, help="已有 metadata.jsonl")
    parser.add_argument("--max_samples", type=int, default=15000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--make_relative", action="store_true", help="路径转为相对 data_root")
    args = parser.parse_args()

    pairs = []
    if args.input_jsonl:
        with open(args.input_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    pairs.append(json.loads(line))
        print(f"Loaded {len(pairs)} pairs from {args.input_jsonl}")
    elif args.stylebooth_dir:
        pairs = find_pairs_stylebooth(args.stylebooth_dir)
        print(f"Found {len(pairs)} pairs from StyleBooth layout")
    elif args.content_dir and args.styled_dir:
        pairs = find_pairs_from_dirs(args.content_dir, args.styled_dir)
        print(f"Found {len(pairs)} pairs from content/styled dirs")
    else:
        # 生成示例 metadata 模板
        print("No input specified. Creating demo metadata template...")
        demo_dir = Path(args.data_root) / "demo"
        demo_dir.mkdir(parents=True, exist_ok=True)
        pairs = [{
            "ref_image": "demo/ref_placeholder.jpg",
            "tgt_image": "demo/tgt_placeholder.jpg",
            "caption": "a photo in artistic style",
            "class": "demo",
        }]
        print(f"Demo template written. Please add images to {demo_dir} and re-run.")

    pairs = subsample(pairs, args.max_samples, args.seed)
    print(f"Final dataset size: {len(pairs)}")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    data_root = Path(args.data_root).resolve()

    with open(args.output, "w", encoding="utf-8") as f:
        for p in pairs:
            entry = dict(p)
            if args.make_relative:
                for key in ("ref_image", "tgt_image"):
                    if key in entry and os.path.isabs(entry[key]):
                        entry[key] = str(Path(entry[key]).relative_to(data_root))
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    # 统计类别数
    classes = set(p.get("class", "default") for p in pairs)
    print(f"Saved {len(pairs)} pairs, {len(classes)} classes -> {args.output}")


if __name__ == "__main__":
    main()
