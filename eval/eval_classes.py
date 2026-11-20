"""在验证集上按类细看：每类召回、平均置信、平均熵，以及最常混的类对。

验证集是按文档隔离的（训练时没见过这些文档），所以这里的召回是真泛化。
"""

import argparse
import os
from collections import Counter, defaultdict

import torch
import torch.nn.functional as F

# 让脚本从子目录里也能找到 src/ 下的模型代码
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import paths

from model import TinyDomainLM


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=paths.ckpt("v2_1h"))
    ap.add_argument("--blocks", default=paths.blocks())
    ap.add_argument("--batch", type=int, default=16)
    args = ap.parse_args()

    ck = torch.load(args.checkpoint, map_location="cpu")
    blob = torch.load(args.blocks, map_location="cpu")
    vocab, domains = ck["vocab"], ck["domains"]
    T = ck["config"]["max_len"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TinyDomainLM(**ck["config"]).to(device)
    model.load_state_dict(ck["model"])
    model.eval()

    docs = (blob["split"] == 1).nonzero().flatten().tolist()
    off = blob["ids"], blob["offsets"]
    per = defaultdict(lambda: {"n": 0, "ok": 0, "conf": 0.0, "ent": 0.0})
    mix = Counter()
    print("checkpoint=%s step=%s 验证文档 %d 篇" %
          (os.path.basename(args.checkpoint), ck.get("step"), len(docs)))

    for start in range(0, len(docs), args.batch):
        chunk = docs[start:start + args.batch]
        xs = []
        for d in chunk:
            s, e = int(off[1][d]), int(off[1][d + 1])
            seq = off[0][s:e].long()
            if len(seq) <= T:
                seq = seq.repeat(-(-T // max(len(seq), 1)))[:T]
            else:
                seq = seq[:T]
            xs.append(seq)
        x = torch.stack(xs).to(device)
        with torch.no_grad():
            logits = model(x, lm_weight=0.0)["domain_logits"]
            p = torch.softmax(logits.float(), dim=-1).cpu()
        for j, d in enumerate(chunk):
            pred = int(p[j].argmax())
            true = int(blob["domain"][d])
            r = per[domains[true]]
            r["n"] += 1
            r["ok"] += int(pred == true)
            r["conf"] += float(p[j][pred])
            r["ent"] += float(-(p[j] * p[j].clamp(min=1e-9).log()).sum())
            if pred != true:
                mix[(domains[true], domains[pred])] += 1

    rows = sorted(per.items(), key=lambda kv: kv[1]["ok"] / max(kv[1]["n"], 1))
    print("\n%-22s %6s %8s %9s %8s" % ("类别", "样本", "召回", "平均置信", "平均熵"))
    for name, r in rows:
        n = max(r["n"], 1)
        print("%-22s %6d %7.1f%% %9.3f %8.3f" %
              (name, r["n"], 100.0 * r["ok"] / n, r["conf"] / n, r["ent"] / n))
    total = sum(r["n"] for r in per.values())
    ok = sum(r["ok"] for r in per.values())
    print("\n总体：%d/%d = %.1f%%" % (ok, total, 100.0 * ok / max(total, 1)))

    print("\n最常混的 10 对（真实 -> 被判成）：")
    for (t, p), c in mix.most_common(10):
        print("  %-20s -> %-20s %4d 次" % (t, p, c))


if __name__ == "__main__":
    main()
