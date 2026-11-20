"""拿训练时没见过的句子试模型，看领域分布是真懂还是只在背训练集。

训练日志里的 99.9% 是训练集内命中率，跑了上百遍，等于背答案。
这个脚本用全新句子做一次独立抽查：
  - 见过的 13 个领域：应该能落到对应格子
  - 没训过的 18 个领域：看它怎么乱猜（这是必然的，别当成功）
  - 顺带看分布是不是软的（熵），以及续写像不像人话
"""

import argparse
import json
import os
import torch

# 让脚本从子目录里也能找到 src/ 下的模型代码
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import paths

from model import TinyDomainLM

TEXTS = [
    ("央行意外下调中期借贷便利利率，A股三大指数早盘集体高开。", "金融_经济"),
    ("下半场补时第九十三分钟，前锋在禁区外一脚远射完成绝杀。", "体育"),
    ("被告未按合同约定交付货物，一审判令其支付违约金及利息。", "法律_司法"),
    ("用 Python 写个装饰器，给函数同时加上缓存和超时控制。", "计算机编程_代码"),
    ("患者术后第三天出现低热，复查血常规提示白细胞计数偏高。", "医学_健康_心理_中医(未训练)"),
    ("今天天气不错，下楼买了杯咖啡，顺路在河边走了半小时。", "日常/其他"),
]

SEEN = {"金融_经济", "体育", "法律_司法", "计算机编程_代码"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=paths.path("runs/10h_20261107/latest.pt"))
    ap.add_argument("--generate", type=int, default=0)
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    ck = torch.load(args.checkpoint, map_location="cpu")
    vocab, domains = ck["vocab"], ck["domains"]
    inv = {v: k for k, v in vocab.items()}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TinyDomainLM(**ck["config"]).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    t = ck.get("temperature", 1.0)
    print("checkpoint=%s  step=%s  温度=%.2f" % (os.path.basename(args.checkpoint), ck.get("step"), t))

    results = []
    for text, expect in TEXTS:
        ids = [vocab.get(c, vocab["<unk>"]) for c in text if not c.isspace()]
        x = torch.tensor([ids], dtype=torch.long, device=device)
        with torch.no_grad():
            logits = model(x, lm_weight=0.0)["domain_logits"]
            p = torch.softmax(logits / t, dim=-1)[0].cpu()
        order = sorted(range(len(domains)), key=lambda i: -p[i].item())
        top = [(domains[i], round(p[i].item(), 4)) for i in order[:3]]
        ent = -(p * p.clamp(min=1e-9).log()).sum().item()
        hit = top[0][0] in expect
        print("\n[%s] 期望=%s" % ("命中" if hit else "未命中", expect))
        print("  %s" % text)
        print("  " + "  ".join("%s %.3f" % (n, v) for n, v in top) + "  熵 %.3f" % ent)
        row = {"text": text, "expect": expect, "hit": hit, "top": top,
               "entropy": round(ent, 4), "dist": {domains[i]: round(p[i].item(), 5) for i in order}}
        if args.generate > 0:
            gen = model.generate(x, max_new=args.generate)
            cont = "".join(inv.get(int(i), "") for i in gen[0]
                           if inv.get(int(i), "") not in {"<pad>", "<unk>", "<bos>"}).replace("<eos>", " / ")
            row["continuation"] = cont
            print("  续写：" + cont[:160])
        results.append(row)

    n_hit = sum(1 for r in results if r["hit"])
    print("\n见过领域的抽查：%d/%d 命中；训练日志的 99.9%% 是训练集内数字，不能当泛化能力。" %
          (n_hit, len([r for r in results if r["expect"] in SEEN])))
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print("已写入 %s" % args.json_out)


if __name__ == "__main__":
    main()
