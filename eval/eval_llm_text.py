"""测「别人写的、不是我们语料文风」的文章。

输入是一个 jsonl，每行 {"domain": 类目, "topic": 题目, "text": 正文}。
分别按几种长度喂给模型，看命中率、置信度、熵，并逐条列出判错的。

  python eval/eval_llm_text.py --file $PRISM_HOME/llm_articles.jsonl
"""

import argparse
import json
import math
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import paths

import torch

from model import TinyDomainLM


def encode(text, vocab):
    unk = vocab["<unk>"]
    return [vocab.get(c, unk) for c in text if not c.isspace()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=paths.ckpt())
    ap.add_argument("--file", default=paths.sample("llm_articles.jsonl"))
    ap.add_argument("--lengths", default="0")   # 0 = 用原文全长（截到 max_len）
    ap.add_argument("--pad", type=int, default=1, help="不足 max_len 是否重复补齐")
    args = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(args.checkpoint, map_location="cpu")
    model = TinyDomainLM(**ck["config"]).to(dev)
    model.load_state_dict(ck["model"])
    model.eval()
    domains, vocab = ck["domains"], ck["vocab"]
    dom_id = {d: i for i, d in enumerate(domains)}
    T = ck["config"]["max_len"]

    rows = [json.loads(l) for l in open(args.file, encoding="utf-8") if l.strip()]
    print("checkpoint=%s step=%s | 文章 %d 篇 | 设备 %s"
          % (os.path.basename(args.checkpoint), ck.get("step"), len(rows), dev))

    lens = [0] if args.lengths == "0" else [int(x) for x in args.lengths.split(",")]

    for L in lens:
        hits, confs, ents, wrong = 0, [], [], []
        for r in rows:
            ids = encode(r["text"], vocab)
            if L:
                ids = ids[:L]
            ids = ids[:T] if L == 0 else ids
            if args.pad and len(ids) < T:
                ids = (ids * ((T + len(ids) - 1) // len(ids)))[:T]
            if not ids:
                continue
            x = torch.tensor([ids], dtype=torch.long, device=dev)
            with torch.no_grad():
                lg = model(x, lm_weight=0.0)["domain_logits"]
                p = torch.softmax(lg.float(), -1)[0].cpu()
            order = sorted(range(len(p)), key=lambda i: -p[i].item())
            pred, true = order[0], dom_id.get(r["domain"], -1)
            conf = float(p[pred])
            ent = float(-(p * p.clamp(min=1e-9).log()).sum())
            confs.append(conf)
            ents.append(ent)
            if pred == true:
                hits += 1
            else:
                wrong.append((r["domain"], domains[pred], float(p[true]), float(p[pred]), r["topic"]))
        n = len(confs)
        tag = "原文长度(pad)" if L == 0 else "前 %d 字" % L
        print("\n=== %s | %d 篇 | 命中 %5.1f%% | 平均置信 %.3f | 平均熵 %.3f ==="
              % (tag, n, 100.0 * hits / max(n, 1), sum(confs) / max(n, 1), sum(ents) / max(n, 1)))
        if wrong:
            print("判错的 %d 条：" % len(wrong))
            for t, p_, pt, pp, topic in wrong:
                print("  %-22s → %-22s (真类得分 %.3f, 判成 %.3f)  %s"
                      % (t, p_, pt, pp, topic))


if __name__ == "__main__":
    main()
