"""启动 10 小时训练前的体检：真跑 forward/backward + 逐配置实测显存与步速。

对应 torch-train-preflight 的 ①②：py_compile 过不等于能跑，
换 batch / 换开关重跑一遍才算数。跑完给出建议配置。
"""

import argparse
import time

import torch

# 让脚本从子目录里也能找到 src/ 下的模型代码
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import paths

from model import TinyDomainLM


def check_forward(cfg, device):
    m = TinyDomainLM(**cfg).to(device)
    B, T = 2, 32
    V, C = cfg["vocab_size"], cfg["n_domains"]
    idx = torch.randint(0, V, (B, T), device=device)
    y = torch.softmax(torch.randn(B, C), dim=-1).to(device)

    m.eval()
    with torch.no_grad():
        o = m(idx, domain_targets=y, lm_weight=1.0, domain_weight=1.0)
        assert o["logits"].shape == (B, T, V), o["logits"].shape
        assert o["domain_logits"].shape == (B, C), o["domain_logits"].shape
        assert o["loss"].ndim == 0 and torch.isfinite(o["loss"]), o["loss"]
        o2 = m(idx, lm_weight=0.0)
        assert o2["domain_logits"].shape == (B, C)
        o3 = m(idx, lm_weight=1.0, domain_weight=0.0)
        assert o3["logits"].shape == (B, T, V)
    print("① forward OK：两条出口 + 三种损失开关，形状与数值全部断言通过")

    m.train()
    out = m(idx, domain_targets=y)
    out["loss"].backward()
    grads = [p.grad for p in m.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    assert abs(out["lm_loss"].item()) > 0 and abs(out["domain_loss"].item()) > 0
    print("   backward OK：两条损失都有梯度，没有死通路")
    del m
    torch.cuda.empty_cache()


def profile(cfg, batch, T, device, warmup=3, reps=10):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    m = TinyDomainLM(**cfg).to(device)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-4)
    scaler = torch.cuda.amp.GradScaler()
    idx = torch.randint(0, cfg["vocab_size"], (batch, T), device=device)
    y = torch.softmax(torch.randn(batch, cfg["n_domains"]), dim=-1).to(device)

    def step():
        opt.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast():
            out = m(idx, domain_targets=y)
        scaler.scale(out["loss"]).backward()
        scaler.step(opt)
        scaler.update()

    for _ in range(warmup):
        step()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(reps):
        step()
    torch.cuda.synchronize()
    dt = (time.time() - t0) / reps
    peak = torch.cuda.max_memory_allocated() / 2**30
    steps_10h = int(10 * 3600 / dt)
    print("② batch=%-3d | %.4f s/step | 峰值 %.2fG | 10h 约 %d 步 | %.0f 字/秒" %
          (batch, dt, peak, steps_10h, batch * T / dt))
    del m, opt, scaler
    torch.cuda.empty_cache()
    return batch, dt, peak, steps_10h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--blocks", default=paths.blocks())
    ap.add_argument("--batches", default="8,16,24")
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    blob = torch.load(args.blocks, map_location="cpu")
    cfg = dict(vocab_size=len(blob["vocab"]), n_domains=len(blob["domains"]),
               d=512, layers=8, heads=8, ff=2048, max_len=blob["block_size"],
               dropout=0.1, pooling="last")
    print("device=%s | 词表 %d | 领域 %d | 长度 %d" %
          (device, cfg["vocab_size"], cfg["n_domains"], cfg["max_len"]))

    check_forward(cfg, device)
    print("② 逐配置实测（warmup 3 + 计时 10 步）：")
    results = []
    for b in [int(x) for x in args.batches.split(",")]:
        try:
            results.append(profile(cfg, b, cfg["max_len"], device))
        except RuntimeError as e:
            print("   batch=%d OOM/失败: %s" % (b, str(e)[:120]))

    free, total = torch.cuda.mem_get_info()
    print("   显存空闲 %.2fG / 总共 %.2fG（桌面进程会随时占用，留 1.5G 边际）" %
          (free / 2**30, total / 2**30))
    usable = [r for r in results if r[2] < 3.0]
    if usable:
        best = max(usable, key=lambda r: r[0])
        print("   建议 batch=%d（峰值 %.2fG，留足边际）" % (best[0], best[2]))


if __name__ == "__main__":
    main()
