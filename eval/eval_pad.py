"""验证用户的发现：短文本把内容重复到 512 字，判断会不会突然变准？

如果「重复补齐」比「原始短句」准得多，说明之前的结论（短文本=信息不够）是错的，
真正的原因是输入长度离开了训练时的分布（训练永远 512 字）。

同一批文档、同一段文字，只改「长度怎么来的」：
  raw30      只喂前 30 字
  rep30      前 30 字重复拼到 512 字
  raw512     直接喂前 512 字（对照）
"""

import argparse

import torch

# 让脚本从子目录里也能找到 src/ 下的模型代码
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import paths

from model import TinyDomainLM


def encode(text, vocab):
    unk = vocab["<unk>"]
    return [vocab.get(c, unk) for c in text if not c.isspace()]


def repeat_to(seq, T):
    if not seq:
        return seq
    return (seq * ((T + len(seq) - 1) // len(seq)))[:T]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=paths.ckpt())
    ap.add_argument("--blocks", default=paths.blocks())
    ap.add_argument("--per-class", type=int, default=4)
    ap.add_argument("--batch", type=int, default=16)
    args = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(args.checkpoint, map_location="cpu")
    blob = torch.load(args.blocks, map_location="cpu")
    domains = ck["domains"]
    T = ck["config"]["max_len"]
    model = TinyDomainLM(**ck["config"]).to(dev)
    model.load_state_dict(ck["model"])
    model.eval()

    ids, off, dom, split = blob["ids"], blob["offsets"], blob["domain"], blob["split"]
    by_cls = {}
    for d in (split == 1).nonzero().flatten().tolist():
        if int(off[d + 1] - off[d]) >= T:
            by_cls.setdefault(int(dom[d]), []).append(d)
    pick = []
    for c in sorted(by_cls):
        pick.extend(by_cls[c][:args.per_class])
    trues = [int(dom[d]) for d in pick]
    print("checkpoint step=%s | %d 篇（%d 类 x %d）| 设备 %s"
          % (ck.get("step"), len(pick), len(by_cls), args.per_class, dev), flush=True)

    def predict(seqs):
        out = []
        for i in range(0, len(seqs), args.batch):
            x = torch.tensor(seqs[i:i + args.batch], dtype=torch.long, device=dev)
            with torch.no_grad():
                lg = model(x, lm_weight=0.0)["domain_logits"]
                out.extend(torch.softmax(lg.float(), -1).cpu().tolist())
        return out

    def score(probs, tag):
        ok = sum(1 for p, t in zip(probs, trues) if max(range(len(p)), key=lambda i: p[i]) == t)
        conf = sum(max(p) for p in probs) / len(probs)
        ent = sum(-sum(q * __import__("math").log(q + 1e-12) for q in p) for p in probs) / len(probs)
        print("  %-28s 命中 %5.1f%%  平均置信 %.3f  平均熵 %.3f"
              % (tag, 100.0 * ok / len(probs), conf, ent), flush=True)
        return ok / len(probs)

    print("\n[对照组] 真实文档，直接用原文前 N 字：")
    for L in (30, 80, 150):
        seqs = []
        for d in pick:
            s = int(off[d])
            seqs.append(ids[s:s + L].long().tolist())
        score(predict(seqs), "原文前 %d 字" % L)

    print("\n[实验组] 同一段文字，重复拼到 512 字：")
    for L in (30, 80, 150):
        seqs = []
        for d in pick:
            s = int(off[d])
            seqs.append(repeat_to(ids[s:s + L].long().tolist(), T))
        score(predict(seqs), "前 %d 字重复到 512" % L)

    print("\n[基准] 完整 512 字：")
    seqs = []
    for d in pick:
        s = int(off[d])
        seqs.append(ids[s:s + T].long().tolist())
    score(predict(seqs), "原文 512 字")

    # 用户的例句
    print("\n[用户例句]")
    vocab = ck["vocab"]
    samples = [
        ("央行意外下调中期借贷便利利率，A股三大指数早盘集体高开。", "金融_经济"),
        ("被告未按合同约定交付货物，一审判令其支付违约金及利息。", "法律_司法"),
    ]
    for text, want in samples:
        base = encode(text, vocab)
        for tag, seq in (("原样 %d 字" % len(base), base),
                         ("重复到 512", repeat_to(base, T))):
            p = predict([seq])[0]
            order = sorted(range(len(p)), key=lambda i: -p[i])[:3]
            print("  %-14s %-10s -> %s" % (tag, want,
                  "  ".join("%s %.1f%%" % (domains[i], p[i] * 100) for i in order)))


if __name__ == "__main__":
    main()
