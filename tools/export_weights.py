"""把训练存档剥成「推理用纯权重」。

训练存档 best.pt 有 382 MB，其中 2/3 是 AdamW 的一阶、二阶矩和随机数状态。
发布时只需要权重本身，而且 fp16 就够——体积降到约 67 MB，能直接放进仓库。

    python tools/export_weights.py --out models/prism-v3-exp.pt

产物结构（推理脚本直接能吃）：

    {"model": state_dict(fp16), "config": {...}, "vocab": {...},
     "domains": [...], "step": 129600, "dtype": "float16"}
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import torch

import paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=paths.ckpt())
    ap.add_argument("--out", default=os.path.join(paths.PROJECT, "models", "prism-v3-exp.pt"))
    ap.add_argument("--fp16", action="store_true", default=True)
    ap.add_argument("--fp32", dest="fp16", action="store_false", help="存 fp32（约 127 MB，超 GitHub 单文件上限）")
    args = ap.parse_args()

    ck = torch.load(args.checkpoint, map_location="cpu")
    sd = ck["model"]
    if args.fp16:
        sd = {k: v.half() for k, v in sd.items()}

    out = {
        "model": sd,
        "config": ck["config"],
        "vocab": ck["vocab"],
        "domains": ck["domains"],
        "step": ck.get("step"),
        "temperature": ck.get("temperature", 1.0),
        "dtype": "float16" if args.fp16 else "float32",
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    torch.save(out, args.out)
    print("写入 %s  %.1f MB  步数 %s  参数 %s"
          % (args.out, os.path.getsize(args.out) / 2 ** 20, out["step"],
             sum(v.numel() for v in sd.values())))


if __name__ == "__main__":
    main()
