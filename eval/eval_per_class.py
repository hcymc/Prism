"""按类算召回：每类均匀抽 N 篇验证文档，看哪几个领域分得最准。

跑 CPU（显存留给训练），2 线程。默认每类 20 篇 = 600 篇，约 5~6 分钟。
"""

import argparse
import time
from collections import Counter, defaultdict

import torch

# 让脚本从子目录里也能找到 src/ 下的模型代码
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import paths

from model import TinyDomainLM


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=paths.ckpt())
    ap.add_argument("--blocks", default=paths.blocks())
    ap.add_argument("--per-class", type=int, default=20)
    ap.add_argument("--batch", type=int, default=10)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    device = torch.device("cpu")

    ck = torch.load(args.checkpoint, map_location="cpu")
    blob = torch.load(args.blocks, map_location="cpu")
    domains = ck["domains"]
    T = ck["config"]["max_len"]
    model = TinyDomainLM(**ck["config"]).to(device)
    model.load_state_dict(ck["model"])
    model.eval()

    ids, off, dom, split = blob["ids"], blob["offsets"], blob["domain"], blob["split"]

    # 每类均匀取 N 篇
    by_cls = defaultdict(list)
    for d in (split == 1).nonzero().flatten().tolist():
        by_cls[int(dom[d])].append(d)
    pick = []
    for c in sorted(by_cls):
        pick.extend(by_cls[c][:args.per_class])
    print("checkpoint=%s | step=%s | 验证文档 %d 篇（%d 类，每类 %d 篇）"
          % (args.checkpoint.split("/")[-1], ck.get("step"), len(pick),
             len(by_cls), args.per_class))

    per = defaultdict(lambda: {"n": 0, "ok": 0, "conf": 0.0})
    mix = Counter()
    t0 = time.time()
    for i in range(0, len(pick), args.batch):
        chunk = pick[i:i + args.batch]
        xs = []
        for d in chunk:
            s, e = int(off[d]), int(off[d + 1])
            seq = ids[s:e].long()
            if len(seq) <= T:
                seq = seq.repeat(-(-T // max(len(seq), 1)))[:T]
            else:
                seq = seq[:T]
            xs.append(seq)
        x = torch.stack(xs)
        with torch.no_grad():
            logits = model(x, lm_weight=0.0)["domain_logits"]
            p = torch.softmax(logits.float(), dim=-1)
        for j, d in enumerate(chunk):
            pred = int(p[j].argmax())
            true = int(dom[d])
            r = per[domains[true]]
            r["n"] += 1
            r["ok"] += int(pred == true)
            r["conf"] += float(p[j][pred])
            if pred != true:
                mix[(domains[true], domains[pred])] += 1

    rows = []
    for name, r in per.items():
        rows.append((name, r["n"], r["ok"] / r["n"], r["conf"] / r["n"]))
    rows.sort(key=lambda x: -x[2])

    print("\n%-3s %-24s %4s %8s %8s" % ("名次", "领域", "篇数", "召回", "平均置信"))
    for i, (name, n, acc, conf) in enumerate(rows, 1):
        mark = " <-- TOP6" if i <= 6 else ""
        print("%-4d %-24s %4d %7.1f%% %8.3f%s" % (i, name, n, acc * 100, conf, mark))

    total_ok = sum(r[2] * r[1] for r in rows)
    total_n = sum(r[1] for r in rows)
    print("\n全类平均（均衡）%.1f%%   最低 %s %.1f%%   最高 %s %.1f%%"
          % (100 * total_ok / total_n, rows[-1][0], rows[-1][2] * 100,
             rows[0][0], rows[0][2] * 100))

    top6 = [r[0] for r in rows[:6]]
    print("\nTOP6 各自最常错的去向：")
    for name in top6:
        bad = [(k[1], v) for k, v in mix.items() if k[0] == name]
        bad.sort(key=lambda x: -x[1])
        err = sum(v for _, v in bad)
        print("  %-22s 错 %2d 篇%s" % (name, err,
              "，主要去向：" + "、".join("%s %d" % b for b in bad[:2]) if bad else "，没错"))

    print("\n耗时 %.1f 分钟" % ((time.time() - t0) / 60))
    if args.json_out:
        import json
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump([{"domain": a, "n": b, "recall": c, "conf": d} for a, b, c, d in rows],
                      f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
