"""看单段文字的完整分布，并对比原样 / 归一化标点两种输入。

  python eval/one_text.py --file $PRISM_HOME/user_sample.txt --top 8
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import paths

import torch

from model import TinyDomainLM
from textproc import encode, to_training_form


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=paths.ckpt())
    ap.add_argument("--file", default=None)
    ap.add_argument("--text", default=None)
    ap.add_argument("--top", type=int, default=8)
    args = ap.parse_args()

    text = args.text
    if args.file:
        text = open(args.file, encoding="utf-8").read()
    if not text:
        raise SystemExit("给 --file 或 --text")

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(args.checkpoint, map_location="cpu")
    model = TinyDomainLM(**ck["config"]).to(dev).eval()
    model.load_state_dict(ck["model"])
    domains, vocab = ck["domains"], ck["vocab"]
    T = ck["config"]["max_len"]
    unk = vocab["<unk>"]

    def run(ids, tag):
        n = len(ids)
        ids = (ids * ((T + len(ids) - 1) // len(ids)))[:T] if len(ids) < T else ids[:T]
        x = torch.tensor([ids], dtype=torch.long, device=dev)
        with torch.no_grad():
            p = torch.softmax(model(x, lm_weight=0.0)["domain_logits"].float(), -1)[0].cpu()
        order = sorted(range(len(p)), key=lambda i: -p[i].item())
        ent = float(-(p * p.clamp(min=1e-9).log()).sum())
        print("\n[%s] 有效字符 %d，截到 %d，分布熵 %.3f" % (tag, n, min(n, T), ent))
        for i in order[:args.top]:
            print("  %-22s %6.2f%%" % (domains[i], 100 * p[i]))
        return order[0], float(p[order[0]])

    raw_ids = [vocab.get(c, unk) for c in text if not c.isspace()]
    n_unk = sum(1 for i in raw_ids if i == unk)
    print("原文 %d 字，其中 %d 个字表里没有（含全角标点）" % (len(raw_ids), n_unk))

    run(raw_ids, "原样喂（全角标点变 <unk>）")
    ids2, dropped = encode(text, vocab)
    run(ids2, "归一化后（全角→半角，丢掉 %d 个生僻字）" % dropped)


if __name__ == "__main__":
    main()
