"""把模型导出成 ONNX，顺便量化成 int8。

服务器只有 2 核 2G、还跑着别的业务，装不了也不该装 torch（1.5G+）。
导出的图只保留领域那一出口，推理时不算 15,341 维的语言头——那一步本来就占 15% 算力。

    python tools/export_onnx.py --out $PRISM_HOME/onnx
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import paths

import torch
import torch.nn as nn
import torch.nn.functional as F

from model import TinyDomainLM, apply_rope


class ExportBlock(nn.Module):
    """导出专用：把 scaled_dot_product_attention 展开成显式的 matmul + mask + softmax。

    数学完全等价，只是 torch 2.2 的 SDPA 在 ONNX 里不好导（会落成自定义算子或者报错）。
    """

    def __init__(self, blk):
        super().__init__()
        self.blk = blk
        self.heads = blk.heads
        self.head_dim = blk.head_dim

    def forward(self, x, cos, sin):
        blk = self.blk
        B, T, D = x.shape
        h = blk.ln1(x)
        qkv = blk.qkv(h).reshape(B, T, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        att = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        mask = torch.full((T, T), float("-inf"), device=x.device).triu(1)
        att = torch.softmax(att + mask, dim=-1)
        out = att @ v
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        x = x + blk.proj(out)
        x = x + blk.mlp(blk.ln2(x))
        return x


class DomainOnly(nn.Module):
    """只走到领域头，不算语言模型头。"""

    def __init__(self, m):
        super().__init__()
        self.max_len = m.max_len
        self.head_dim = m.head_dim
        self.tok = m.tok
        self.blocks = nn.ModuleList([ExportBlock(b) for b in m.blocks])
        self.ln_f = m.ln_f
        self.pool_score = m.pool_score
        self.domain_head = m.domain_head

    def forward(self, idx):
        from model import rope_cache
        B, T = idx.shape
        cos, sin = rope_cache(T, self.head_dim, idx.device)
        h = self.tok(idx)
        for blk in self.blocks:
            h = blk(h, cos, sin)
        h = self.ln_f(h)
        pooled = h[:, -1]
        return self.domain_head(pooled)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=paths.ckpt())
    ap.add_argument("--out", default=paths.path("onnx"))
    ap.add_argument("--quantize", action="store_true", default=True)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    ck = torch.load(args.checkpoint, map_location="cpu")
    model = TinyDomainLM(**ck["config"]).eval()
    model.load_state_dict(ck["model"])
    T = ck["config"]["max_len"]

    dm = DomainOnly(model).eval()
    x = torch.randint(0, ck["config"]["vocab_size"], (1, T), dtype=torch.long)

    with torch.no_grad():
        ref = dm(x)
        t0 = time.perf_counter()
        for _ in range(5):
            dm(x)
        torch_ms = (time.perf_counter() - t0) / 5 * 1000

    fp32 = os.path.join(args.out, "domain_fp32.onnx")
    torch.onnx.export(
        dm, (x,), fp32,
        input_names=["ids"], output_names=["logits"],
        opset_version=17, do_constant_folding=True,
    )
    print("导出 fp32: %.1f MB, torch 单次 %.0f ms" % (os.path.getsize(fp32) / 2 ** 20, torch_ms))

    if args.quantize:
        from onnxruntime.quantization import QuantType, quantize_dynamic
        int8 = os.path.join(args.out, "domain_int8.onnx")
        quantize_dynamic(fp32, int8, weight_type=QuantType.QInt8, per_channel=True)
        print("量化 int8: %.1f MB" % (os.path.getsize(int8) / 2 ** 20))

    # 对一下一致性
    import numpy as np
    import onnxruntime as ort

    for path in [fp32] + ([os.path.join(args.out, "domain_int8.onnx")] if args.quantize else []):
        so = ort.SessionOptions()
        so.intra_op_num_threads = 1
        so.log_severity_level = 3
        sess = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
        arr = x.numpy()
        out = sess.run(None, {"ids": arr})[0]
        d = float(np.abs(out - ref.numpy()).max())
        # 单线程计时
        t0 = time.perf_counter()
        for _ in range(10):
            sess.run(None, {"ids": arr})
        ms = (time.perf_counter() - t0) / 10 * 1000
        print("%-28s 与 torch 最大差 %.4f  1 线程单次 %.0f ms"
              % (os.path.basename(path), d, ms))

    np.save(os.path.join(args.out, "sample_ids.npy"), x.numpy())


if __name__ == "__main__":
    main()
