"""
A small, modern decoder-only transformer (Llama-style), written from scratch.

What's modern here, versus the textbook GPT-2 block:
  * RoPE (rotary position embeddings) instead of learned absolute positions
  * RMSNorm instead of LayerNorm
  * SwiGLU feed-forward instead of GELU MLP
  * Grouped-query attention (GQA): fewer key/value heads than query heads,
    which shrinks the KV cache
  * No bias terms, pre-norm blocks, tied input/output embeddings
  * Residual-branch init scaled by 1/sqrt(2 * n_layer)
  * KV-cache inference with temperature / top-k / top-p / repetition penalty
"""
import math
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    vocab_size: int = 4096
    block_size: int = 256
    n_layer: int = 8
    n_head: int = 6
    n_kv_head: int = 2          # GQA: 3 query heads share each key/value head
    n_embd: int = 384
    ffn_hidden: int = 1024      # SwiGLU hidden size (~ 8/3 * n_embd, rounded)
    rope_theta: float = 10000.0
    norm_eps: float = 1e-5

    def __post_init__(self):
        assert self.n_embd % self.n_head == 0, "n_embd must be divisible by n_head"
        assert self.n_head % self.n_kv_head == 0, "n_head must be divisible by n_kv_head"
        assert (self.n_embd // self.n_head) % 2 == 0, "head_dim must be even for RoPE"


# --------------------------------------------------------------------------
# building blocks
# --------------------------------------------------------------------------
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        xf = x.float()
        out = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return out.type_as(x) * self.weight


def build_rope(head_dim, max_len, theta):
    """Precompute cos/sin tables of shape (max_len, head_dim/2)."""
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    t = torch.arange(max_len).float()
    freqs = torch.outer(t, inv_freq)
    return freqs.cos(), freqs.sin()


def apply_rope(x, cos, sin):
    """x: (B, heads, T, head_dim); cos/sin: (T, head_dim/2). Rotates pairs of dims."""
    half = x.size(-1) // 2
    x1, x2 = x[..., :half], x[..., half:]
    cos, sin = cos[None, None], sin[None, None]
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


class Attention(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.n_head, self.n_kv_head = cfg.n_head, cfg.n_kv_head
        self.head_dim = cfg.n_embd // cfg.n_head
        self.wq = nn.Linear(cfg.n_embd, cfg.n_head * self.head_dim, bias=False)
        self.wk = nn.Linear(cfg.n_embd, cfg.n_kv_head * self.head_dim, bias=False)
        self.wv = nn.Linear(cfg.n_embd, cfg.n_kv_head * self.head_dim, bias=False)
        self.wo = nn.Linear(cfg.n_head * self.head_dim, cfg.n_embd, bias=False)

    def forward(self, x, cos, sin, past_kv=None):
        B, T, _ = x.shape
        assert past_kv is None or T == 1, "with a KV cache, feed one token at a time"
        q = self.wq(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)

        if past_kv is not None:
            k = torch.cat([past_kv[0], k], dim=2)
            v = torch.cat([past_kv[1], v], dim=2)
        new_kv = (k, v)  # cached with only n_kv_head heads (the point of GQA)

        rep = self.n_head // self.n_kv_head
        if rep > 1:  # expand kv heads to match query heads
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)

        y = F.scaled_dot_product_attention(q, k, v, is_causal=(past_kv is None))
        y = y.transpose(1, 2).contiguous().view(B, T, -1)
        return self.wo(y), new_kv


class SwiGLU(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.w_gate = nn.Linear(cfg.n_embd, cfg.ffn_hidden, bias=False)
        self.w_up = nn.Linear(cfg.n_embd, cfg.ffn_hidden, bias=False)
        self.w_down = nn.Linear(cfg.ffn_hidden, cfg.n_embd, bias=False)

    def forward(self, x):
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


class Block(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.n_embd, cfg.norm_eps)
        self.attn = Attention(cfg)
        self.ffn_norm = RMSNorm(cfg.n_embd, cfg.norm_eps)
        self.ffn = SwiGLU(cfg)

    def forward(self, x, cos, sin, past_kv=None):
        a, kv = self.attn(self.attn_norm(x), cos, sin, past_kv)
        x = x + a
        x = x + self.ffn(self.ffn_norm(x))
        return x, kv


# --------------------------------------------------------------------------
# sampling
# --------------------------------------------------------------------------
def sample_logits(logits, history, temperature=0.8, top_k=40, top_p=0.95,
                  repetition_penalty=1.1, rep_window=64):
    """logits: (B, vocab) for the next token; history: (B, T) token ids so far."""
    logits = logits.float()
    if repetition_penalty != 1.0:
        prev = history[:, -rep_window:]
        score = torch.gather(logits, 1, prev)
        score = torch.where(score < 0, score * repetition_penalty, score / repetition_penalty)
        logits = logits.scatter(1, prev, score)
    if temperature <= 0:
        return logits.argmax(dim=-1, keepdim=True)  # greedy
    logits = logits / temperature
    if top_k and top_k > 0:
        k = min(top_k, logits.size(-1))
        kth = torch.topk(logits, k).values[:, [-1]]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    if top_p and top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True)
        probs = F.softmax(sorted_logits, dim=-1)
        remove = (probs.cumsum(dim=-1) - probs) > top_p  # always keeps the top token
        sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
        logits = torch.full_like(logits, float("-inf")).scatter(1, sorted_idx, sorted_logits)
    return torch.multinomial(F.softmax(logits, dim=-1), 1)


# --------------------------------------------------------------------------
# the model
# --------------------------------------------------------------------------
class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.norm = RMSNorm(cfg.n_embd, cfg.norm_eps)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight  # weight tying

        cos, sin = build_rope(cfg.n_embd // cfg.n_head, cfg.block_size, cfg.rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        self.apply(self._init_weights)
        # scale residual-branch output projections so activations don't grow with depth
        for name, p in self.named_parameters():
            if name.endswith("wo.weight") or name.endswith("w_down.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))

    @staticmethod
    def _init_weights(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def num_params(self):
        return sum(p.numel() for p in self.parameters())

    # ---- training / full-sequence forward ----
    def forward(self, idx, targets=None):
        B, T = idx.shape
        assert T <= self.cfg.block_size, f"sequence {T} > block_size {self.cfg.block_size}"
        x = self.tok_emb(idx)
        cos, sin = self.rope_cos[:T], self.rope_sin[:T]
        for blk in self.blocks:
            x, _ = blk(x, cos, sin)
        logits = self.lm_head(self.norm(x))
        if targets is None:
            return logits, None
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    # ---- inference with KV cache; returns logits for the LAST position only ----
    def forward_cached(self, idx, past_kvs=None, start_pos=0):
        B, T = idx.shape
        assert start_pos + T <= self.cfg.block_size
        x = self.tok_emb(idx)
        cos = self.rope_cos[start_pos : start_pos + T]
        sin = self.rope_sin[start_pos : start_pos + T]
        new_kvs = []
        for i, blk in enumerate(self.blocks):
            x, kv = blk(x, cos, sin, None if past_kvs is None else past_kvs[i])
            new_kvs.append(kv)
        return self.lm_head(self.norm(x[:, -1, :])), new_kvs

    @torch.no_grad()
    def generate_stream(self, idx, max_new_tokens, temperature=0.8, top_k=40, top_p=0.95,
                        repetition_penalty=1.1, rep_window=64, stop_id=None):
        """Yields one token id at a time, using the KV cache."""
        was_training = self.training
        self.eval()
        block = self.cfg.block_size
        kvs, cached = None, 0
        try:
            for _ in range(max_new_tokens):
                if kvs is None or cached >= block:
                    # prefill the prompt (or rebuild when the context window is full)
                    ctx = idx[:, -block:] if kvs is None else idx[:, -(block // 2):]
                    logits, kvs = self.forward_cached(ctx)
                    cached = ctx.size(1)
                else:
                    logits, kvs = self.forward_cached(idx[:, -1:], kvs, cached)
                    cached += 1
                nxt = sample_logits(logits, idx, temperature, top_k, top_p,
                                    repetition_penalty, rep_window)
                idx = torch.cat([idx, nxt], dim=1)
                tid = nxt.item()
                yield tid
                if stop_id is not None and tid == stop_id:
                    break
        finally:
            self.train(was_training)

    @torch.no_grad()
    def generate_nocache(self, idx, max_new_tokens, temperature=0.8, top_k=40, top_p=0.95,
                         repetition_penalty=1.1, rep_window=64, stop_id=None):
        """Reference implementation (recomputes everything each step). For tests/benchmarks."""
        was_training = self.training
        self.eval()
        try:
            for _ in range(max_new_tokens):
                logits, _ = self(idx[:, -self.cfg.block_size:])
                nxt = sample_logits(logits[:, -1, :], idx, temperature, top_k, top_p,
                                    repetition_penalty, rep_window)
                idx = torch.cat([idx, nxt], dim=1)
                if stop_id is not None and nxt.item() == stop_id:
                    break
        finally:
            self.train(was_training)
        return idx

    @classmethod
    def from_checkpoint(cls, path, device="cpu"):
        ck = torch.load(path, map_location=device)
        model = cls(GPTConfig(**ck["config"]))
        model.load_state_dict(ck["model"])
        return model.to(device).eval(), ck

    def config_dict(self):
        return asdict(self.cfg)
