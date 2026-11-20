"""全角/半角标点对判断准确率的影响——控制变量实验。

同一批文字，只改标点形态，别的都不变：

  raw      原样（全角标点 → <unk>，也就是现在的线上行为）
  ascii    全角换成半角（还原成训练语料的样子），查不到的字仍塞 <unk>
  ascii*   全角换半角 + 查不到的字直接丢掉
  strip    标点全部删掉，只留字

两个数据源：
  --source llm     DeepSeek 写的 30 篇（真有全角标点）
  --source valid   验证集文档（本来就没有全角标点，把半角改成全角来模拟真实文本）

  python eval/eval_punct.py --source llm  --file $PRISM_HOME/llm_articles_news.jsonl
  python eval/eval_punct.py --source valid --docs 120
"""

import argparse
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import paths

import torch

from model import TinyDomainLM
from textproc import to_training_form

PUNCT = "，。、；：？！“”‘’（）《》【】「」—…·"
HALF2FULL = {",": "，", ".": "。", '"': "“", ";": "；", ":": "：",
             "?": "？", "!": "！", "(": "（", ")": "）"}


def to_fullwidth(text):
    """把半角标点改回全角，模拟从网页复制来的真实文本。"""
    return "".join(HALF2FULL.get(c, c) for c in text)


def strip_punct(text):
    drop = set(PUNCT) | set(",.\";:?!()<>[]'")
    return "".join(c for c in text if c not in drop)


def enc_raw(text, vocab):
    unk = vocab["<unk>"]
    return [vocab.get(c, unk) for c in text if not c.isspace()]


def enc_map(text, vocab, drop_unknown=False):
    unk = vocab["<unk>"]
    out = []
    for ch in to_training_form(text):
        if ch.isspace():
            continue
        i = vocab.get(ch)
        if i is None:
            if drop_unknown:
                continue
            out.append(unk)
        else:
            out.append(i)
    return out


VARIANTS = {
    "raw":     lambda t, v: enc_raw(t, v),
    "ascii":   lambda t, v: enc_map(t, v, drop_unknown=False),
    "ascii*":  lambda t, v: enc_map(t, v, drop_unknown=True),
    "strip":   lambda t, v: enc_map(strip_punct(t), v, drop_unknown=True),
}


def load_llm(path):
    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    return [(r["text"], r["domain"]) for r in rows]


def load_valid(path, n, T):
    blob = torch.load(path, map_location="cpu")
    ids, off, dom, split, names = (blob["ids"], blob["offsets"], blob["domain"],
                                   blob["split"], blob["domains"])
    inv = {i: c for c, i in blob["vocab"].items()}
    by = defaultdict(list)
    for d in (split == 1).nonzero().flatten().tolist():
        by[int(dom[d])].append(d)
    per = max(1, n // max(len(by), 1))
    out = []
    for c in sorted(by):
        for d in by[c][:per]:
            s = int(off[d])
            seq = ids[s:s + T].long().tolist()
            text = "".join(inv.get(i, "") for i in seq)
            if len(text) >= 100:
                out.append((text, names[int(dom[d])]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=paths.ckpt())
    ap.add_argument("--blocks", default=paths.blocks())
    ap.add_argument("--source", default="llm", choices=["llm", "valid"])
    ap.add_argument("--file", default=paths.sample("llm_articles_news.jsonl"))
    ap.add_argument("--docs", type=int, default=120)
    ap.add_argument("--trunc", type=int, default=0, help="只取前 N 字（模拟短文本）")
    args = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(args.checkpoint, map_location="cpu")
    model = TinyDomainLM(**ck["config"]).to(dev).eval()
    model.load_state_dict(ck["model"])
    domains, vocab = ck["domains"], ck["vocab"]
    dom_id = {d: i for i, d in enumerate(domains)}
    T = ck["config"]["max_len"]

    if args.source == "llm":
        data = load_llm(args.file)
        print("数据源：%s（%d 篇）" % (os.path.basename(args.file), len(data)))
    else:
        data = load_valid(args.blocks, args.docs, T)
        # 验证集原文没有全角标点，人为改回全角，模拟从网页复制的文本
        data = [(to_fullwidth(t), d) for t, d in data]
        print("数据源：验证集 %d 篇（原文的半角标点已改成全角）" % len(data))

    def predict(ids):
        if len(ids) < 8:
            return None
        ids = (ids * ((T + len(ids) - 1) // len(ids)))[:T] if len(ids) < T else ids[:T]
        x = torch.tensor([ids], dtype=torch.long, device=dev)
        with torch.no_grad():
            lg = model(x, lm_weight=0.0)["domain_logits"]
            p = torch.softmax(lg.float(), -1)[0]
        pred = int(p.argmax())
        ent = float(-(p * p.clamp(min=1e-9).log()).sum())
        return pred, float(p[pred]), ent

    print("\n%-8s %6s %8s %8s %8s" % ("标点处理", "样本", "命中率", "平均置信", "平均熵"))
    print("-" * 44)
    base = None
    for name, fn in VARIANTS.items():
        ok = n = 0
        conf = ent = 0.0
        for text, dom in data:
            t = text[:args.trunc] if args.trunc else text
            r = predict(fn(t, vocab))
            if r is None:
                continue
            pred, c, e = r
            n += 1
            conf += c
            ent += e
            if pred == dom_id.get(dom, -1):
                ok += 1
        if not n:
            continue
        acc = ok / n
        if base is None:
            base = acc
        mark = "" if base is None else ("  %+.1f pt" % ((acc - base) * 100) if name != "raw" else "")
        print("%-8s %6d %7.1f%% %8.3f %8.3f%s"
              % (name, n, 100 * acc, conf / n, ent / n, mark))


if __name__ == "__main__":
    main()
