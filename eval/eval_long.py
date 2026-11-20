"""长文本评测：每类取 N 篇 2000 字以上的文档，一次读不完就切块平均。

模型最大只吃 512 字，所以「读 2000 字」的做法是：切成 4 段 512 字，分别判，再把概率平均。
同时给出「只取前 512 字」的对照，看多读这 1500 字到底有没有用。
"""

import argparse
import json
import time
from collections import defaultdict

import torch

# 让脚本从子目录里也能找到 src/ 下的模型代码
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import paths

from model import TinyDomainLM

T = 512
NCHUNK = 4  # 4 x 512 = 2048 字


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=paths.ckpt())
    ap.add_argument("--blocks", default=paths.blocks())
    ap.add_argument("--per-class", type=int, default=8)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--json-out", default=paths.path("runs/v3_30class/eval_long.json"))
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    ck = torch.load(args.checkpoint, map_location="cpu")
    blob = torch.load(args.blocks, map_location="cpu")
    domains = ck["domains"]
    model = TinyDomainLM(**ck["config"]).to("cpu")
    model.load_state_dict(ck["model"])
    model.eval()

    ids, off, dom, split = blob["ids"], blob["offsets"], blob["domain"], blob["split"]
    need = T * NCHUNK

    by_cls = defaultdict(list)
    for d in (split == 1).nonzero().flatten().tolist():
        if int(off[d + 1] - off[d]) >= need:
            by_cls[int(dom[d])].append(d)
    pick = []
    for c in sorted(by_cls):
        pick.extend(by_cls[c][:args.per_class])
    print("checkpoint=%s | step=%s | 取 %d 篇（%d 类，每类 %d 篇，每篇 >= %d 字）"
          % (args.checkpoint.split("/")[-1], ck.get("step"), len(pick),
             len(by_cls), args.per_class, need), flush=True)

    def run(chunk_idx):
        """对每篇取第 chunk_idx 个 512 字块，返回 [probs]。"""
        out = []
        for i in range(0, len(pick), args.batch):
            xs = []
            for d in pick[i:i + args.batch]:
                s = int(off[d]) + chunk_idx * T
                xs.append(ids[s:s + T].long())
            x = torch.stack(xs)
            with torch.no_grad():
                lg = model(x, lm_weight=0.0)["domain_logits"]
                out.extend(torch.softmax(lg.float(), -1).tolist())
        return out

    t0 = time.time()
    chunks = []
    for k in range(NCHUNK):
        chunks.append(run(k))
        print("  第 %d/4 块算完  %.1f 分钟" % (k + 1, (time.time() - t0) / 60), flush=True)

    trues = [int(dom[d]) for d in pick]

    def score(prob_list, tag):
        per = defaultdict(lambda: {"n": 0, "ok": 0, "conf": 0.0})
        ok_all = 0
        for p, t in zip(prob_list, trues):
            pred = max(range(len(p)), key=lambda i: p[i])
            r = per[domains[t]]
            r["n"] += 1
            r["ok"] += int(pred == t)
            r["conf"] += p[pred]
            ok_all += int(pred == t)
        out = {k: {"n": v["n"], "acc": v["ok"] / v["n"], "conf": v["conf"] / v["n"]}
               for k, v in per.items()}
        return {"tag": tag, "acc": ok_all / len(prob_list), "per": out}

    # A：只看前 512 字
    a = score(chunks[0], "前512字")
    # B：4 块概率平均
    avg = [[sum(chunks[k][i][j] for k in range(NCHUNK)) / NCHUNK
            for j in range(len(chunks[0][0]))] for i in range(len(pick))]
    b = score(avg, "2048字切块平均")

    for s in (a, b):
        print("\n=== %s | 全类均衡平均 %.1f%% ===" % (s["tag"], s["acc"] * 100))

    order = sorted(a["per"], key=lambda k: -a["per"][k]["acc"])
    print("\n%-3s %-24s %4s %10s %10s %6s" % ("#", "领域", "篇数", "前512字", "2048平均", "变化"))
    for i, name in enumerate(order, 1):
        ra, rb = a["per"][name], b["per"][name]
        delta = (rb["acc"] - ra["acc"]) * 100
        print("%-3d %-24s %4d %9.1f%% %9.1f%% %6s"
              % (i, name, ra["n"], ra["acc"] * 100, rb["acc"] * 100,
                 ("%+.0f" % delta) if abs(delta) > 0.5 else "—"))

    print("\n耗时 %.1f 分钟" % ((time.time() - t0) / 60))
    with open(args.json_out, "w", encoding="utf-8") as f:
        json.dump({"checkpoint": args.checkpoint, "step": ck.get("step"),
                   "per_class": args.per_class, "chars": need,
                   "first512": a["acc"], "avg2048": b["acc"],
                   "rows": [{"domain": name, "n": a["per"][name]["n"],
                             "first512": a["per"][name]["acc"],
                             "avg2048": b["per"][name]["acc"]} for name in order]},
                  f, ensure_ascii=False, indent=1)
    print("写入 %s" % args.json_out)


if __name__ == "__main__":
    main()
