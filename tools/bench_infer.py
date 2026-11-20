"""量一下推理到底吃多少资源：单次延迟、内存峰值、线程数影响。

    python tools/bench_infer.py --device cpu
    python tools/bench_infer.py --device cpu --threads 1
    python tools/bench_infer.py --device cuda
"""

import argparse
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import paths

import torch

from model import TinyDomainLM
from textproc import encode


def peak_mem_mb():
    """当前进程的峰值内存（Windows 用 GetProcessMemoryInfo，Linux 读 /proc）。"""
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        class PMC(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                        ("PeakWorkingSetSize", ctypes.c_size_t),
                        ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t),
                        ("PeakPagefileUsage", ctypes.c_size_t)]

        pmc = PMC()
        pmc.cb = ctypes.sizeof(PMC)
        ctypes.windll.psapi.GetProcessMemoryInfo(
            ctypes.windll.kernel32.GetCurrentProcess(), ctypes.byref(pmc), pmc.cb)
        return pmc.PeakWorkingSetSize / 2 ** 20
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmHWM:"):
                    return int(line.split()[1]) / 1024
    except OSError:
        pass
    return 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=paths.ckpt())
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--runs", type=int, default=20)
    args = ap.parse_args()

    if args.threads:
        torch.set_num_threads(args.threads)
    dev = torch.device(args.device)

    t0 = time.time()
    ck = torch.load(args.checkpoint, map_location="cpu")
    model = TinyDomainLM(**ck["config"]).to(dev).eval()
    model.load_state_dict(ck["model"])
    load_s = time.time() - t0
    vocab, T = ck["vocab"], ck["config"]["max_len"]
    n_param = sum(p.numel() for p in model.parameters())

    sample = ("央行今日下调中期借贷便利利率十个基点，市场普遍预期后续还有降准空间。"
              "分析人士认为，此次调整主要针对实体经济融资成本，对债券市场形成支撑。") * 6
    ids, _ = encode(sample, vocab)
    ids = (ids * ((T + len(ids) - 1) // len(ids)))[:T]
    x = torch.tensor([ids], dtype=torch.long, device=dev)

    for _ in range(3):
        with torch.no_grad():
            model(x, lm_weight=0.0)

    ts = []
    for _ in range(args.runs):
        t = time.perf_counter()
        with torch.no_grad():
            model(x, lm_weight=0.0)
        ts.append((time.perf_counter() - t) * 1000)

    print("设备 %s  torch %s  线程 %d" % (dev, torch.__version__, torch.get_num_threads()))
    print("参数 %.2f 万  权重 fp32 约 %.0f MB" % (n_param / 1e4, n_param * 4 / 2 ** 20))
    print("加载耗时 %.2f s" % load_s)
    print("512 字单次：中位 %.1f ms  均值 %.1f ms  P95 %.1f ms  最快 %.1f ms"
          % (statistics.median(ts), statistics.mean(ts),
             sorted(ts)[max(0, int(len(ts) * 0.95) - 1)], min(ts)))
    print("进程峰值内存 %.0f MB" % peak_mem_mb())


if __name__ == "__main__":
    main()
