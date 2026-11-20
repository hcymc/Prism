"""跨快照对比：领域头是在什么时候从「真会分」变成「背答案」的。

训练日志的 99.9% 是训练集内命中率，跑上百遍等于背答案。
这里用同一批没见过的句子，测四个时间点的独立表现：
命中数 / 平均最高权重 / 平均熵（31 类完全均匀时熵约 3.43）。
"""

import os
import torch

from eval_unseen import TEXTS, SEEN
# 让脚本从子目录里也能找到 src/ 下的模型代码
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import paths

from model import TinyDomainLM

RUN = paths.path("runs", "10h_20261107")
CKPTS = [
    ("2.5万步", "ckpt_step0025000.pt"),
    ("10万步", "ckpt_step0100000.pt"),
    ("20万步", "ckpt_step0200000.pt"),
    ("29.6万步", "latest.pt"),
]


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("快照".ljust(12), "见过领域命中", "平均最高权重", "平均熵", "  (熵越大越像瞎摊)")
    for name, fn in CKPTS:
        path = os.path.join(RUN, fn)
        if not os.path.isfile(path):
            print(name, "缺文件", fn)
            continue
        ck = torch.load(path, map_location="cpu")
        vocab, domains = ck["vocab"], ck["domains"]
        cfg = dict(ck["config"])
        # 10h 那轮的权重用的是可学习位置编码，config 里没有 pos_type 字段，按权重推断
        if "pos.weight" in ck["model"]:
            cfg["pos_type"] = "learned"
        model = TinyDomainLM(**cfg).to(device)
        model.load_state_dict(ck["model"])
        model.eval()
        t = ck.get("temperature", 1.0)
        hits, tops, ents = 0, [], []
        for text, expect in TEXTS:
            ids = [vocab.get(c, vocab["<unk>"]) for c in text if not c.isspace()]
            x = torch.tensor([ids], dtype=torch.long, device=device)
            with torch.no_grad():
                logits = model(x, lm_weight=0.0)["domain_logits"]
                p = torch.softmax(logits / t, dim=-1)[0].cpu()
            top_i = int(p.argmax().item())
            top_p = float(p[top_i].item())
            ent = -(p * p.clamp(min=1e-9).log()).sum().item()
            tops.append(top_p)
            ents.append(ent)
            if expect in SEEN and domains[top_i] in expect:
                hits += 1
        print(name.ljust(12), ("%d/4" % hits).ljust(12),
              ("%.3f" % (sum(tops) / len(tops))).ljust(12),
              "%.3f" % (sum(ents) / len(ents)))
        del model, ck
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
