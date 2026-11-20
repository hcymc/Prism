"""用训好的模型：给一段文字，吐出 31 个格子各占几成；也能让它续写两句。"""

import argparse
import json
import os

import torch

import paths
from model import TinyDomainLM
from textproc import encode as tp_encode



def load(path, device):
    ckpt = torch.load(path, map_location="cpu")
    model = TinyDomainLM(**ckpt["config"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return ckpt, model


def encode(text, vocab):
    # 全角标点归一化 + 丢掉字表外的字，和训练语料对齐（详见 textproc.py）
    return tp_encode(text, vocab)[0]


def pad_to(ids, T):
    """短文本重复补齐到 T：训练时永远见 T 字，短输入属于没见过的形态。"""
    if not ids or len(ids) >= T:
        return ids[:T]
    return (ids * ((T + len(ids) - 1) // len(ids)))[:T]


def decode(seq, inv):
    skip = {"<pad>", "<unk>", "<bos>"}
    return "".join(inv.get(int(i), "") for i in seq if inv.get(int(i), "") not in skip).replace("<eos>", " / ")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=paths.ckpt(), help="权重路径，默认 $PRISM_HOME/runs/v3_30class/best.pt")
    ap.add_argument("--text", default="关于金融_经济的一点记录，先看背景，再看过程。")
    ap.add_argument("--topk", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--generate", type=int, default=0, help="续写多少字，0 表示不续写")
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    if not os.path.isfile(args.checkpoint):
        raise SystemExit("找不到 %s，先跑 train.py" % args.checkpoint)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt, model = load(args.checkpoint, device)
    vocab, domains = ckpt["vocab"], ckpt["domains"]

    ids = encode(args.text, vocab)
    if not ids:
        raise SystemExit("输入文本编码后是空的")
    ids = pad_to(ids, ckpt["config"]["max_len"])
    x = torch.tensor([ids], dtype=torch.long, device=device)
    t = args.temperature * ckpt.get("temperature", 1.0)

    with torch.no_grad():
        dom_logits = model(x, lm_weight=0.0)["domain_logits"]
        probs = torch.softmax(dom_logits / t, dim=-1)[0].cpu().tolist()
    order = sorted(range(len(domains)), key=lambda i: -probs[i])
    print("领域分布（温度 %.2f）：" % t)
    for i in order[:args.topk]:
        print("  %-22s %.4f  (%.1f%%)" % (domains[i], probs[i], probs[i] * 100))
    print("合计 %.4f" % sum(probs))

    result = {"text": args.text, "dist": {domains[i]: round(probs[i], 5) for i in order}}

    inv = {v: k for k, v in vocab.items()}
    if args.generate > 0:
        gen = model.generate(x, max_new=args.generate)
        result["continuation"] = decode(gen[0], inv)
        print("续写：%s" % result["continuation"][:200])

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print("已写入 %s" % args.json_out)


if __name__ == "__main__":
    main()
