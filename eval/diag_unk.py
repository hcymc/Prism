"""诊断：LLM 文章在模型眼里长什么样——未知字比例、前三个预测、文本开头。"""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import paths

import torch

from model import TinyDomainLM

CK = paths.ckpt()
F = sys.argv[1] if len(sys.argv) > 1 else paths.sample("llm_articles_news.jsonl")

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ck = torch.load(CK, map_location="cpu")
model = TinyDomainLM(**ck["config"]).to(dev).eval()
model.load_state_dict(ck["model"])
domains, vocab = ck["domains"], ck["vocab"]
unk = vocab["<unk>"]
dom_id = {d: i for i, d in enumerate(domains)}
T = ck["config"]["max_len"]

rows = [json.loads(l) for l in open(F, encoding="utf-8") if l.strip()]
for r in rows:
    ids = [vocab.get(c, unk) for c in r["text"] if not c.isspace()][:T]
    n_unk = sum(1 for i in ids if i == unk)
    if len(ids) < T:
        ids = (ids * ((T + len(ids) - 1) // len(ids)))[:T]
    x = torch.tensor([ids], dtype=torch.long, device=dev)
    with torch.no_grad():
        p = torch.softmax(model(x, lm_weight=0.0)["domain_logits"].float(), -1)[0].cpu()
    order = sorted(range(len(p)), key=lambda i: -p[i].item())[:3]
    print("%-20s 长%3d 未知%3d(%.1f%%) -> %s" %
          (r["domain"], len(list(r["text"])), n_unk, 100.0 * n_unk / max(len(ids), 1),
           "  ".join("%s %.2f" % (domains[i], p[i]) for i in order)))

print("\n--- 第一条正文开头 200 字 ---")
print(rows[0]["text"][:200])
