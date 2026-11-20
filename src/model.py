"""一个从零训练的小语言模型，共用一副身子、两个出口。

出口 A（lm_head）：照常猜下一个字。这才是「预训练」——纯文本自己就是老师。
出口 B（domain_head）：整段读完之后，从池化后的状态里吐出 31 个格子各占几成。

为什么要两个：只留 B，模型就不学说话了，也拿不到自监督信号，
那就只是穿了语言模型外壳的分类器。留住 A，它才真的是个 LM。

池化默认用「最后一个字的状态」：
这类模型读字是单向的，前面的字看不见后面的，取平均会把一堆半瞎的
状态混进来；最后那个字看完了整段，信息最全。--pooling 可切换。

位置编码两种：
  learned（旧）—— 可学习位置嵌入，长度一变就得重训，外推差；
  rope（新）    —— 旋转位置编码，不加参数，512 长度下更稳，也方便以后拉长上下文。
"""

import torch
import torch.nn.functional as F
from torch import nn


def rope_cache(T, head_dim, device, base=10000.0):
    inv = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(T, device=device).float()
    freqs = torch.outer(t, inv)
    return freqs.cos(), freqs.sin()


def apply_rope(x, cos, sin):
    """x: [B, H, T, Dh]；cos/sin: [T, Dh/2]，按奇偶维旋转。"""
    x1, x2 = x[..., 0::2], x[..., 1::2]
    c = cos[None, None, :, :]
    s = sin[None, None, :, :]
    o1 = x1 * c - x2 * s
    o2 = x1 * s + x2 * c
    return torch.stack([o1, o2], dim=-1).flatten(-2)


class Block(nn.Module):
    def __init__(self, d, heads, ff, dropout):
        super().__init__()
        self.heads = heads
        self.head_dim = d // heads
        self.ln1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.ln2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(
            nn.Linear(d, ff), nn.GELU(), nn.Linear(ff, d), nn.Dropout(dropout)
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x, cos=None, sin=None):
        B, T, D = x.shape
        h = self.ln1(x)
        qkv = self.qkv(h).reshape(B, T, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        if cos is not None:
            q = apply_rope(q, cos, sin)
            k = apply_rope(k, cos, sin)
        out = F.scaled_dot_product_attention(
            q, k, v, is_causal=True,
            dropout_p=self.drop.p if self.training else 0.0,
        )
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        x = x + self.drop(self.proj(out))
        x = x + self.mlp(self.ln2(x))
        return x


class TinyDomainLM(nn.Module):
    def __init__(self, vocab_size, n_domains, d=512, layers=8, heads=8,
                 ff=2048, max_len=512, dropout=0.1, pooling="last",
                 pos_type="rope"):
        super().__init__()
        self.max_len = max_len
        self.pooling = pooling
        self.pos_type = pos_type
        self.head_dim = d // heads
        self.tok = nn.Embedding(vocab_size, d)
        self.pos = nn.Embedding(max_len, d) if pos_type == "learned" else None
        self.blocks = nn.ModuleList(
            [Block(d, heads, ff, dropout) for _ in range(layers)]
        )
        self.ln_f = nn.LayerNorm(d)
        self.lm_head = nn.Linear(d, vocab_size, bias=False)
        self.lm_head.weight = self.tok.weight  # 词表头和输入共享权重，省参数
        self.pool_score = nn.Linear(d, 1)
        self.domain_head = nn.Sequential(
            nn.Linear(d, d), nn.GELU(), nn.Linear(d, n_domains)
        )
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def pool(self, h):
        if self.pooling == "last":
            return h[:, -1]
        if self.pooling == "mean":
            return h.mean(1)
        if self.pooling == "attn":
            w = torch.softmax(self.pool_score(h).squeeze(-1), dim=-1)
            return (h * w.unsqueeze(-1)).sum(1)
        raise ValueError("pooling 只能是 last / mean / attn")

    def encode(self, idx):
        """只跑身子，返回每个位置的隐状态。"""
        B, T = idx.shape
        if T > self.max_len:
            raise ValueError("序列长度 %d 超过 max_len %d" % (T, self.max_len))
        if self.pos_type == "rope":
            cos, sin = rope_cache(T, self.head_dim, idx.device)
            h = self.tok(idx)
            for blk in self.blocks:
                h = blk(h, cos, sin)
        else:
            pos = torch.arange(T, device=idx.device)
            h = self.tok(idx) + self.pos(pos)[None, :, :]
            for blk in self.blocks:
                h = blk(h)
        return self.ln_f(h)

    def forward(self, idx, domain_targets=None, lm_weight=1.0, domain_weight=1.0):
        h = self.encode(idx)
        logits = self.lm_head(h)
        pooled = self.pool(h)
        dom_logits = self.domain_head(pooled)
        out = {"logits": logits, "domain_logits": dom_logits, "pooled": pooled}

        loss = torch.zeros((), device=idx.device)
        if lm_weight > 0:
            lm_loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, logits.size(-1)).float(),
                idx[:, 1:].reshape(-1),
            )
            out["lm_loss"] = lm_loss
            loss = loss + lm_weight * lm_loss
        if domain_targets is not None and domain_weight > 0:
            dom_loss = -(domain_targets * F.log_softmax(dom_logits.float(), -1)).sum(-1).mean()
            out["domain_loss"] = dom_loss
            loss = loss + domain_weight * dom_loss
        out["loss"] = loss
        return out

    @torch.no_grad()
    def generate(self, idx, max_new=48, temperature=0.9, top_k=40):
        self.eval()
        for _ in range(max_new):
            cond = idx[:, -self.max_len:]
            logits = self(cond, lm_weight=0.0)["logits"][:, -1, :] / temperature
            if top_k and top_k > 0:
                cut = torch.topk(logits, min(top_k, logits.size(-1)))[0][:, [-1]]
                logits = logits.masked_fill(logits < cut, float("-inf"))
            nxt = torch.multinomial(torch.softmax(logits, dim=-1), 1)
            idx = torch.cat([idx, nxt], dim=1)
        return idx
