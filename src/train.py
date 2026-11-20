"""从零训练：一个身子，多个出口 + 文档级监督 + 独立验证集 + 早停。

针对 11-07 那次 10 小时长训暴露的问题做的改动：
  1. 验证集与训练集**按文档**隔离（prepare.py 负责），这里每 eval-every 步评一次；
  2. 早停：验证集域 KL 连续 patience 次没改善就停，保存 best.pt；
  3. 文档级监督：同一篇文档取**两个不同位置的块**，两个都得对上同一个答案；
  4. 一致性损失：两块的输出分布互相靠拢（对称 KL），逼模型学语义而不是记字串；
  5. 对比损失：同篇两块互为正样本、batch 内其他为负，用池化表示做 InfoNCE；
  6. 位置编码换 RoPE，块长默认 512（更完整的上下文）。

损失 = lm_w * 猜字 + dom_w * 领域 + cons_w * 一致性 + ct_w * 对比
"""

import argparse
import json
import math
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from model import TinyDomainLM
import paths

HERE = os.path.dirname(os.path.abspath(__file__))


def lr_at(step, total, warmup, base):
    if step < warmup:
        return base * (step + 1) / max(warmup, 1)
    prog = (step - warmup) / max(total - warmup, 1)
    return base * (0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * prog)))


def decode(seq, inv):
    skip = {"<pad>", "<unk>", "<bos>"}
    return "".join(inv.get(int(i), "") for i in seq if inv.get(int(i), "") not in skip).replace("<eos>", " / ")


def emit(run_dir, msg):
    print(msg, flush=True)
    with open(os.path.join(run_dir, "train.log"), "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def write_heartbeat(run_dir, data):
    p = os.path.join(run_dir, "heartbeat.json")
    tmp = p + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, p)


class DocDataset(Dataset):
    """文档级：每篇取两个不同位置的块（训练）或固定两端（验证）。"""

    def __init__(self, blob, split, block_size, train, limit=0, seed=0):
        self.ids = blob["ids"]
        self.offsets = blob["offsets"]
        self.dist = blob["dist"]
        docs = (blob["split"] == split).nonzero().flatten().tolist()
        if limit and limit < len(docs):
            rng = random.Random(seed)
            rng.shuffle(docs)          # 抽样前打散，保证每个领域都有
            docs = docs[:limit]
        self.docs = docs
        self.T = block_size
        self.train = train

    def set_block(self, T):
        """换块长做长度增广：一个 batch 内必须同长，所以按批次整批切换。"""
        self.T = int(T)

    def __len__(self):
        return len(self.docs)

    def __getitem__(self, i):
        d = self.docs[i]
        s, e = int(self.offsets[d]), int(self.offsets[d + 1])
        seq = self.ids[s:e].long()
        T = self.T
        if len(seq) <= T:
            a = seq.repeat(math.ceil((T + 1) / max(len(seq), 1)))[:T]
            b = a
        elif self.train:
            s1 = random.randint(0, len(seq) - T)
            s2 = random.randint(0, len(seq) - T)
            for _ in range(3):
                if abs(s1 - s2) >= T // 2:
                    break
                s2 = random.randint(0, len(seq) - T)
            a = seq[s1:s1 + T]
            b = seq[s2:s2 + T]
        else:
            a = seq[:T]
            b = seq[-T:]
        return a, b, self.dist[d]


def build_loader(blob, split, block_size, batch, train):
    ds = DocDataset(blob, split, block_size, train)
    return DataLoader(ds, batch_size=batch, shuffle=train, drop_last=train,
                      num_workers=0), ds


@torch.no_grad()
def evaluate(model, loader, device, use_fp16, temp=1.0):
    model.eval()
    tot = {"loss": 0.0, "lm": 0.0, "dom": 0.0, "acc": 0.0, "ent": 0.0, "kl": 0.0, "ppl": 0.0}
    n = 0
    eps = 1e-9
    for a, _b, y in loader:
        a, y = a.to(device), y.to(device)
        with torch.cuda.amp.autocast(enabled=use_fp16):
            out = model(a, domain_targets=y)
        p = torch.softmax(out["domain_logits"].float() / temp, dim=-1)
        tot["loss"] += out["loss"].item()
        tot["lm"] += out["lm_loss"].item()
        tot["dom"] += out["domain_loss"].item()
        tot["ppl"] += math.exp(min(out["lm_loss"].item(), 20.0))
        tot["acc"] += (p.argmax(-1) == y.argmax(-1)).float().mean().item()
        tot["ent"] += -(p * p.clamp(min=eps).log()).sum(-1).mean().item()
        tot["kl"] += (y * (y.clamp(min=eps).log() - p.clamp(min=eps).log())).sum(-1).mean().item()
        n += 1
    model.train()
    return {k: v / max(n, 1) for k, v in tot.items()}


def save(run_dir, name, model, cfg, vocab, domains, args, opt, scaler, step, wall, extra=None):
    state = {
        "model": model.state_dict(), "config": cfg, "vocab": vocab, "domains": domains,
        "step": step, "wall_seconds": wall, "args": vars(args),
        "opt": opt.state_dict(), "scaler": scaler.state_dict(),
        "torch_rng": torch.get_rng_state(), "temperature": 1.0,
    }
    if extra:
        state.update(extra)
    path = os.path.join(run_dir, name)
    torch.save(state, path)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--blocks", default=paths.blocks())
    ap.add_argument("--run-dir", default=None)
    ap.add_argument("--resume", default=None)
    ap.add_argument("--mode", default="both", choices=["both", "lm", "domain"])
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--max-hours", type=float, default=0.0)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--d", type=int, default=512)
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--ff", type=int, default=2048)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--pooling", default="last", choices=["last", "mean", "attn"])
    ap.add_argument("--pos-type", default="rope", choices=["rope", "learned"])
    ap.add_argument("--block-size", type=int, default=0, help="0 表示用 blocks.pt 里的")
    ap.add_argument("--min-block", type=int, default=0, help="长度增广下界，0 表示用 blocks.pt 里的")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--lm-weight", type=float, default=1.0)
    ap.add_argument("--domain-weight", type=float, default=2.0)
    ap.add_argument("--cons-weight", type=float, default=0.5)
    ap.add_argument("--ct-weight", type=float, default=0.1)
    ap.add_argument("--ct-temp", type=float, default=0.1)
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--valid-limit", type=int, default=1500, help="验证集抽样篇数，0=全部")
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--sample-every", type=int, default=5000)
    ap.add_argument("--save-every", type=int, default=1000)
    ap.add_argument("--milestone-every", type=int, default=0)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--threads", type=int, default=2, help="CPU 线程数；GPU 训练时 2 个就够，多了白占 CPU")
    ap.add_argument("--no-fp16", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_fp16 = device.type == "cuda" and not args.no_fp16
    # GPU 训练时 CPU 只负责取数和小算子，默认线程池会把 6 个核全占满，白烧 CPU
    try:
        torch.set_num_threads(max(1, args.threads))
        torch.set_num_interop_threads(1)
    except Exception:
        pass

    run_dir = args.run_dir or "$PRISM_HOME/runs/" + time.strftime("%Y%m%d_%H%M%S")
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "train.log"), "a", encoding="utf-8") as f:
        f.write("run_dir=%s start=%s\n" % (run_dir, time.strftime("%Y-%m-%d %H:%M:%S")))
    emit(run_dir, "args=%s" % json.dumps(vars(args), ensure_ascii=False))

    blob = torch.load(args.blocks, map_location="cpu")
    vocab, domains = blob["vocab"], blob["domains"]
    block_size = args.block_size or blob["block_size"]
    cfg = dict(vocab_size=len(vocab), n_domains=len(domains), d=args.d,
               layers=args.layers, heads=args.heads, ff=args.ff,
               max_len=block_size, dropout=args.dropout, pooling=args.pooling,
               pos_type=args.pos_type)
    model = TinyDomainLM(**cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.1)
    scaler = torch.cuda.amp.GradScaler(enabled=use_fp16)
    start_step = 0
    if args.resume and os.path.isfile(args.resume):
        ck = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        scaler.load_state_dict(ck["scaler"])
        torch.set_rng_state(ck["torch_rng"])
        start_step = ck["step"]
        emit(run_dir, "从 %s 续训，起始 step=%d" % (args.resume, start_step))

    train_loader, train_ds = build_loader(blob, 0, block_size, args.batch, True)
    valid_ds = DocDataset(blob, 1, block_size, False, limit=args.valid_limit)
    valid_loader = DataLoader(valid_ds, batch_size=args.batch, shuffle=False, num_workers=0)
    min_block = args.min_block or blob.get("min_block", block_size)
    min_block = min(int(min_block), block_size)
    span = block_size - min_block
    block_grid = sorted({block_size - round(span * i / 4) for i in range(5)}) if span > 0 else [block_size]
    n_params = sum(p.numel() for p in model.parameters())
    emit(run_dir, "device=%s fp16=%s | 参数 %.2f 万 | 块长 %d | batch %d | 训练 %d 篇 / 验证 %d 篇" %
         (device, use_fp16, n_params / 1e4, block_size, args.batch, len(train_ds), len(valid_ds)))

    lm_w = 0.0 if args.mode == "domain" else args.lm_weight
    dom_w = 0.0 if args.mode == "lm" else args.domain_weight
    it = iter(train_loader)
    model.train()
    wall0 = time.time()
    t0 = wall0
    hb = {"run_dir": run_dir, "max_hours": args.max_hours, "alive": True}
    keys = ("lm", "dom", "cons", "ct", "total", "acc", "ent", "kl")
    running = {k: 0.0 for k in keys}
    last_step = start_step
    best_kl, best_step, bad = 1e9, start_step, 0

    for step in range(start_step + 1, args.steps + 1):
        # 长度增广：每 400 步换一次块长（256~512），让模型见过不同上下文长度
        if len(block_grid) > 1 and step % 400 == 0:
            train_ds.set_block(random.choice(block_grid))
        try:
            a, b, y = next(it)
        except StopIteration:
            it = iter(train_loader)
            a, b, y = next(it)
        a, b, y = a.to(device), b.to(device), y.to(device)

        lr = lr_at(step, args.steps, args.warmup, args.lr)
        for g in opt.param_groups:
            g["lr"] = lr
        opt.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=use_fp16):
            out_a = model(a, domain_targets=y, lm_weight=lm_w, domain_weight=dom_w)
            out_b = model(b, domain_targets=y, lm_weight=lm_w, domain_weight=dom_w)
            base = 0.5 * (out_a["loss"] + out_b["loss"])
            la = F.log_softmax(out_a["domain_logits"].float(), -1)
            lb = F.log_softmax(out_b["domain_logits"].float(), -1)
            cons = 0.5 * (F.kl_div(la, lb.exp(), reduction="batchmean") +
                          F.kl_div(lb, la.exp(), reduction="batchmean"))
            za = F.normalize(out_a["pooled"].float(), dim=-1)
            zb = F.normalize(out_b["pooled"].float(), dim=-1)
            z = torch.cat([za, zb], 0)
            sim = z @ z.t() / args.ct_temp
            sim.masked_fill_(torch.eye(len(z), dtype=torch.bool, device=device), -1e4)
            B = za.size(0)
            labels = torch.cat([torch.arange(B, device=device) + B,
                                torch.arange(B, device=device)])
            ct = F.cross_entropy(sim, labels)
            loss = base + args.cons_weight * cons + args.ct_weight * ct
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()

        with torch.no_grad():
            p = torch.softmax(out_a["domain_logits"].float(), -1)
            eps = 1e-9
            vals = {
                "lm": out_a["lm_loss"].item() if "lm_loss" in out_a else 0.0,
                "dom": out_a["domain_loss"].item() if "domain_loss" in out_a else 0.0,
                "cons": cons.item(), "ct": ct.item(), "total": loss.item(),
                "acc": (p.argmax(-1) == y.argmax(-1)).float().mean().item(),
                "ent": -(p * p.clamp(min=eps).log()).sum(-1).mean().item(),
                "kl": (y * (y.clamp(min=eps).log() - p.clamp(min=eps).log())).sum(-1).mean().item(),
            }
        for k in keys:
            running[k] += vals[k]
        last_step = step

        if step % args.log_every == 0 or step == start_step + 1:
            n = args.log_every if step != start_step + 1 else 1
            dt = time.time() - t0
            avg = {k: running[k] / n for k in keys}
            emit(run_dir,
                 "step %6d | %.0f min | loss %.4f (字 %.3f / 域 %.3f / 一致 %.3f / 对比 %.3f) | "
                 "acc %.3f | ent %.2f | kl %.3f | lr %.2e | %.0f 字/秒 | 显存 %.2fG" %
                 (step, (time.time() - wall0) / 60, avg["total"], avg["lm"], avg["dom"],
                  avg["cons"], avg["ct"], avg["acc"], avg["ent"], avg["kl"], lr,
                  (n * args.batch * block_size) / max(dt, 1e-6),
                  torch.cuda.max_memory_allocated() / 2**30 if device.type == "cuda" else 0))
            hb.update({"step": step, "wall_seconds": round(time.time() - wall0, 1),
                       "train_loss": round(avg["total"], 4), "train_acc": round(avg["acc"], 4),
                       "train_dom": round(avg["dom"], 4), "cons": round(avg["cons"], 4),
                       "ct": round(avg["ct"], 4), "lr": lr,
                       "vram_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2)
                       if device.type == "cuda" else 0,
                       "updated": time.strftime("%Y-%m-%d %H:%M:%S"), "alive": True})
            write_heartbeat(run_dir, hb)
            for k in keys:
                running[k] = 0.0
            t0 = time.time()

        if step % args.eval_every == 0:
            m = evaluate(model, valid_loader, device, use_fp16)
            flag = ""
            if m["kl"] < best_kl - 1e-4:
                best_kl, best_step, bad = m["kl"], step, 0
                save(run_dir, "best.pt", model, cfg, vocab, domains, args, opt, scaler, step,
                     time.time() - wall0, {"valid": m})
                flag = "  <- best"
            else:
                bad += 1
            emit(run_dir,
                 "  [验证] step %6d | loss %.4f | 域 %.3f | acc %.3f | ent %.2f | kl %.3f | "
                 "ppl %.1f | best %.3f@%d | 连续未改善 %d%s" %
                 (step, m["loss"], m["dom"], m["acc"], m["ent"], m["kl"], m["ppl"],
                  best_kl, best_step, bad, flag))
            hb.update({
                "step": step, "wall_seconds": round(time.time() - wall0, 1),
                "train_loss": round(m["loss"], 4),
                "valid_acc": round(m["acc"], 4), "valid_ent": round(m["ent"], 4),
                "valid_kl": round(m["kl"], 4), "valid_ppl": round(m["ppl"], 2),
                "best_kl": round(best_kl, 4), "best_step": best_step,
                "bad_evals": bad, "vram_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2)
                if device.type == "cuda" else 0,
                "updated": time.strftime("%Y-%m-%d %H:%M:%S"), "alive": True,
            })
            write_heartbeat(run_dir, hb)
            if args.patience > 0 and bad >= args.patience:
                emit(run_dir, "连续 %d 次验证没改善，早停（best step=%d，kl=%.4f）" %
                     (bad, best_step, best_kl))
                break

        if args.milestone_every > 0 and step % args.milestone_every == 0:
            save(run_dir, "ckpt_step%07d.pt" % step, model, cfg, vocab, domains, args,
                 opt, scaler, step, time.time() - wall0)
        if args.save_every > 0 and step % args.save_every == 0:
            save(run_dir, "latest.pt", model, cfg, vocab, domains, args, opt, scaler,
                 step, time.time() - wall0)

        if args.sample_every > 0 and step % args.sample_every == 0 and args.mode != "domain":
            model.eval()
            prompt = "关于金融经济的一点记录"
            p = torch.tensor([[vocab.get(c, vocab["<unk>"]) for c in prompt]],
                             dtype=torch.long, device=device)
            gen = model.generate(p, max_new=40)
            emit(run_dir, "  续写：%s" % decode(gen[0], {v: k for k, v in vocab.items()})[:120])
            model.train()

        if args.max_hours > 0 and (time.time() - wall0) >= args.max_hours * 3600:
            emit(run_dir, "达到 %.1f 小时上限" % args.max_hours)
            break

    save(run_dir, "ckpt_final.pt", model, cfg, vocab, domains, args, opt, scaler,
         last_step, time.time() - wall0)
    emit(run_dir, "结束：step=%d，最佳验证 step=%d（kl=%.4f），用时 %.2f 小时" %
         (last_step, best_step, best_kl, (time.time() - wall0) / 3600))
    write_heartbeat(run_dir, {"step": last_step, "alive": False, "best_step": best_step,
                              "best_kl": round(best_kl, 4), "run_dir": run_dir,
                              "finished": time.strftime("%Y-%m-%d %H:%M:%S")})


if __name__ == "__main__":
    main()
