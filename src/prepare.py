"""把语料打包成「文档级」数据集：按文档切分训练/验证，块在训练时在线随机切。

为什么要这样改（对照 11-07 那次 10 小时长训的教训）：
  旧做法把所有文本拼成一条流、固定切成 256 字的块，train/valid 不分离。
  结果模型把 49762 块背了 143 遍，训练集领域命中 99.9%，没见过的句子只命中 1~2/6。

新做法：
  1. 每篇文档独立存 id 序列，**按文档**分训练/验证，同一篇的块绝不会跨两侧（防泄漏）；
  2. 训练时每次随机取起点切片，每个 epoch 看到的切分都不同，记不住字串；
  3. 存成「一大块 int16 + offsets」，加载快、占空间小（6000 万字约 120MB）。

输出 blocks.pt：
  ids      int16   [总字数]   所有文档拼在一起
  offsets  int32   [N+1]      第 i 篇是 ids[offsets[i]:offsets[i+1]]
  dist     float32 [N, 31]    每篇的领域答案
  split    uint8   [N]        0=train 1=valid
  domain   int16   [N]        领域下标
  vocab / domains / block_size / stats
"""

import argparse
import json
import os
import random
from collections import Counter

import torch

from domains import DOMAINS, INDEX, N
import paths

HERE = os.path.dirname(os.path.abspath(__file__))
PAD, UNK, BOS, EOS = "<pad>", "<unk>", "<bos>", "<eos>"
SPECIAL = [PAD, UNK, BOS, EOS]

TEMPLATES = [
    "关于{t}的一点记录，先看背景，再看过程，最后把结论写清楚。",
    "今天整理{t}的材料，顺手把几条要点记下来，以后大概还会用到。",
    "{t}这个方向最近变化不少，值得单独留出一段来说一说。",
    "有人问到{t}，我把手头能找到的说法粗略汇总了一遍。",
    "一篇偏{t}的随笔，没有定论，只是把过程写得明白些。",
    "工作日志：今天处理的事情主要和{t}有关，另外记了几个数字。",
    "翻到一段讲{t}的旧材料，读完之后觉得有些地方可以再核一核。",
    "{t}相关的条目又多了一批，先按时间排好，等有空再逐条看。",
]


def read_docs(root):
    """返回 {领域名: [文档文本, ...]}，一篇文档一条记录。"""
    buckets = {}
    for name in sorted(os.listdir(root)):
        p = os.path.join(root, name)
        if not os.path.isdir(p) or name not in INDEX:
            continue
        texts = []
        for fn in sorted(os.listdir(p)):
            fp = os.path.join(p, fn)
            if not os.path.isfile(fp):
                continue
            if fn.endswith(".jsonl"):
                with open(fp, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            t = (json.loads(line).get("text") or "").strip()
                        except json.JSONDecodeError:
                            continue
                        if t:
                            texts.append(t)
            elif fn.endswith(".txt"):
                with open(fp, "r", encoding="utf-8") as f:
                    t = f.read().strip()
                    if t:
                        texts.append(t)
        if texts:
            buckets[name] = texts
    return buckets


def make_demo(seed=7, per_domain=300):
    rng = random.Random(seed)
    buckets = {}
    for name in DOMAINS:
        topic = name.replace("_", "、")
        buckets[name] = [
            rng.choice(TEMPLATES).format(t=topic) + "第%d条。" % (i + 1)
            for i in range(per_domain)
        ]
    return buckets


def build_vocab(buckets, vocab_size):
    cnt = Counter()
    for texts in buckets.values():
        for t in texts:
            cnt.update(ch for ch in t if not ch.isspace())
    vocab = {s: i for i, s in enumerate(SPECIAL)}
    for c, _ in cnt.most_common(max(vocab_size - len(SPECIAL), 1)):
        vocab[c] = len(vocab)
    return vocab


def encode(text, vocab):
    out = []
    for ch in text:
        if ch == "\n":
            out.append(vocab[EOS])
        elif ch.isspace():
            continue
        else:
            out.append(vocab.get(ch, vocab[UNK]))
    return out


def make_dist(name, smooth):
    d = [0.0] * N
    d[INDEX[name]] = 1.0
    if smooth > 0:
        smooth = min(max(smooth, 0.0), 0.9)
        d = [x * (1.0 - smooth) for x in d]
        for i in range(N):
            if i != INDEX[name]:
                d[i] += smooth / (N - 1)
    s = sum(d)
    return [x / s for x in d]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", default=None)
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--out", default=paths.blocks())
    ap.add_argument("--vocab-out", default=paths.vocab())
    ap.add_argument("--vocab-size", type=int, default=16000)
    ap.add_argument("--block-size", type=int, default=512)
    ap.add_argument("--min-block", type=int, default=256, help="训练时随机块长的下界")
    ap.add_argument("--valid-ratio", type=float, default=0.1)
    ap.add_argument("--min-len", type=int, default=16, help="短于此字数的文档丢弃")
    ap.add_argument("--smooth", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    if args.demo or not args.input_dir:
        buckets = make_demo(seed=args.seed)
        source = "demo"
    else:
        if not os.path.isdir(args.input_dir):
            raise SystemExit("找不到目录: %s" % args.input_dir)
        buckets = read_docs(args.input_dir)
        source = args.input_dir
    if not buckets:
        raise SystemExit("没读到任何语料")

    vocab = build_vocab(buckets, args.vocab_size)
    ids_list, dist_list, split_list, dom_list, offsets = [], [], [], [], [0]
    stat = {}
    for name, texts in buckets.items():
        docs = [t for t in texts if len(t) >= args.min_len]
        rng.shuffle(docs)
        n_valid = max(1, int(len(docs) * args.valid_ratio)) if len(docs) > 4 else 0
        valid_idx = set(range(n_valid))
        dist = make_dist(name, args.smooth)
        chars = 0
        for i, t in enumerate(docs):
            seq = encode(t, vocab)
            seq.append(vocab[EOS])
            if len(seq) < 8:
                continue
            ids_list.append(torch.tensor(seq, dtype=torch.int16))
            offsets.append(offsets[-1] + len(seq))
            dist_list.append(dist)
            split_list.append(1 if i in valid_idx else 0)
            dom_list.append(INDEX[name])
            chars += len(seq)
        stat[name] = {"docs": len(docs), "valid": n_valid, "chars": chars}

    if not dist_list:
        raise SystemExit("没有可用文档")

    ids = torch.cat(ids_list)
    blob = {
        "ids": ids,
        "offsets": torch.tensor(offsets, dtype=torch.int32),
        "dist": torch.tensor(dist_list, dtype=torch.float32),
        "split": torch.tensor(split_list, dtype=torch.uint8),
        "domain": torch.tensor(dom_list, dtype=torch.int16),
        "vocab": vocab,
        "domains": DOMAINS,
        "block_size": args.block_size,
        "min_block": args.min_block,
        "source": source,
        "stats": stat,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save(blob, args.out)
    with open(args.vocab_out, "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False)

    n_train = int((blob["split"] == 0).sum())
    n_valid = int((blob["split"] == 1).sum())
    print("领域 %d 个 | 文档 %d 篇（训练 %d / 验证 %d）| 词表 %d | 总字数 %d | 块长 %d" %
          (len(stat), len(dist_list), n_train, n_valid, len(vocab), len(ids), args.block_size))
    for name, s in sorted(stat.items(), key=lambda x: -x[1]["chars"])[:8]:
        print("  %-22s %5d 篇 / 验证 %3d / %8d 字" % (name, s["docs"], s["valid"], s["chars"]))
    print("已写入 %s（按文档切分，同一篇的块只落在一侧）" % args.out)


if __name__ == "__main__":
    main()
