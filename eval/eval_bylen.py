"""同一批文档、只改输入长度，看领域判断随长度怎么变。

固定「长度 ≥ 512」的验证文档，对每篇取前 20/50/100/200/350/512 字，
分别预测。这样样本、标签、领域分布都不变，唯一变量就是长度。
"""

import argparse
import os
from collections import defaultdict

import torch

# 让脚本从子目录里也能找到 src/ 下的模型代码
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import paths

from model import TinyDomainLM


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=paths.ckpt("v2_1h"))
    ap.add_argument("--blocks", default=paths.blocks())
    ap.add_argument("--lengths", default="20,50,100,200,350,512")
    ap.add_argument("--batch", type=int, default=24)
    ap.add_argument("--max-docs", type=int, default=1200)
    args = ap.parse_args()

    ck = torch.load(args.checkpoint, map_location="cpu")
    blob = torch.load(args.blocks, map_location="cpu")
    domains = ck["domains"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TinyDomainLM(**ck["config"]).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    maxlen = ck["config"]["max_len"]

    ids, off = blob["ids"], blob["offsets"]
    valid = (blob["split"] == 1).nonzero().flatten().tolist()
    docs = [d for d in valid if int(off[d + 1]) - int(off[d]) >= 512][:args.max_docs]
    print("checkpoint=%s step=%s | 取长度>=512 的验证文档 %d 篇" %
          (os.path.basename(args.checkpoint), ck.get("step"), len(docs)))

    lens = [int(x) for x in args.lengths.split(",")]
    stats = defaultdict(lambda: {"n": 0, "ok": 0, "conf": 0.0, "ent": 0.0})

    for L in lens:
        L = min(L, maxlen)
        for start in range(0, len(docs), args.batch):
            chunk = docs[start:start + args.batch]
            xs = []
            for d in chunk:
                s = int(off[d])
                xs.append(ids[s:s + L].long())
            x = torch.stack(xs).to(device)
            with torch.no_grad():
                logits = model(x, lm_weight=0.0)["domain_logits"]
                p = torch.softmax(logits.float(), dim=-1).cpu()
            for j, d in enumerate(chunk):
                pred = int(p[j].argmax())
                true = int(blob["domain"][d])
                r = stats[L]
                r["n"] += 1
                r["ok"] += int(pred == true)
                r["conf"] += float(p[j][pred])
                r["ent"] += float(-(p[j] * p[j].clamp(min=1e-9).log()).sum())

    print("\n%-8s %7s %8s %10s %8s" % ("输入字数", "样本", "命中率", "平均置信", "平均熵"))
    for L in sorted(stats):
        r = stats[L]
        n = max(r["n"], 1)
        print("%-8d %7d %7.1f%% %10.3f %8.3f" %
              (L, r["n"], 100.0 * r["ok"] / n, r["conf"] / n, r["ent"] / n))


if __name__ == "__main__":
    main()
