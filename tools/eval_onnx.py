"""对一下 ONNX（fp32 / int8）和 torch 原模型的命中率，别让量化悄悄吃掉准确率。

    python tools/eval_onnx.py --docs 300
"""

import argparse
import os
import sys
import time
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import paths

import numpy as np
import torch

from model import TinyDomainLM
from textproc import encode


def load_docs(blocks, n_per_class, T):
    blob = torch.load(blocks, map_location="cpu")
    ids, off, dom, split, names = (blob["ids"], blob["offsets"], blob["domain"],
                                   blob["split"], blob["domains"])
    inv = {i: c for c, i in blob["vocab"].items()}
    by = defaultdict(list)
    for d in (split == 1).nonzero().flatten().tolist():
        by[int(dom[d])].append(d)
    out = []
    for c in sorted(by):
        for d in by[c][:n_per_class]:
            s = int(off[d])
            text = "".join(inv.get(i, "") for i in ids[s:s + T].long().tolist())
            if len(text) >= 100:
                out.append((text, names[int(dom[d])]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=paths.ckpt())
    ap.add_argument("--blocks", default=paths.blocks())
    ap.add_argument("--onnx-dir", default=paths.path("onnx"))
    ap.add_argument("--docs", type=int, default=300)
    args = ap.parse_args()

    ck = torch.load(args.checkpoint, map_location="cpu")
    T, vocab = ck["config"]["max_len"], ck["vocab"]
    domain_names = ck["domains"]
    name2id = {d: i for i, d in enumerate(domain_names)}

    docs = load_docs(args.blocks, max(1, args.docs // 30), T)
    print("样本 %d 篇" % len(docs))

    X = []
    for text, _ in docs:
        ids, _ = encode(text, vocab)
        ids = (ids * ((T + len(ids) - 1) // len(ids)))[:T] if len(ids) < T else ids[:T]
        X.append(ids)
    X = np.array(X, dtype=np.int64)
    y = np.array([name2id.get(d, -1) for _, d in docs])

    results = {}

    model = TinyDomainLM(**ck["config"]).eval()
    model.load_state_dict(ck["model"])
    t0 = time.perf_counter()
    with torch.no_grad():
        lg = model(torch.from_numpy(X), lm_weight=0.0)["domain_logits"].numpy()
    ms = (time.perf_counter() - t0) / len(X) * 1000
    results["torch fp32"] = (lg, ms)

    import onnxruntime as ort
    for tag, fname in (("onnx fp32", "domain_fp32.onnx"), ("onnx int8", "domain_int8.onnx")):
        path = os.path.join(args.onnx_dir, fname)
        if not os.path.isfile(path):
            continue
        so = ort.SessionOptions()
        so.intra_op_num_threads = 1
        so.log_severity_level = 3
        sess = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
        outs = []
        t0 = time.perf_counter()
        for i in range(len(X)):
            outs.append(sess.run(None, {"ids": X[i:i + 1]})[0][0])
        ms = (time.perf_counter() - t0) / len(X) * 1000
        results[tag] = (np.array(outs), ms)

    base = None
    print("\n%-12s %8s %10s %10s %8s" % ("模型", "命中率", "与torch同判", "平均置信", "单篇ms"))
    print("-" * 54)
    for tag, (lg, ms) in results.items():
        p = np.exp(lg - lg.max(-1, keepdims=True))
        p /= p.sum(-1, keepdims=True)
        acc = float((lg.argmax(-1) == y).mean())
        agree = float((lg.argmax(-1) == results["torch fp32"][0].argmax(-1)).mean())
        conf = float(p.max(-1).mean())
        if base is None:
            base = acc
        print("%-12s %7.1f%% %9.1f%% %9.3f %8.0f %s"
              % (tag, acc * 100, agree * 100, conf, ms,
                 "" if tag == "torch fp32" else ("%+.1f pt" % ((acc - base) * 100))))


if __name__ == "__main__":
    main()
