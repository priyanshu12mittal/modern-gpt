"""Run with:  python -m pytest -q   (or just: python test_model.py)"""
import math
import torch
from model import GPT, GPTConfig, apply_rope, build_rope

torch.manual_seed(0)
TINY = GPTConfig(vocab_size=64, block_size=32, n_layer=2, n_head=4, n_kv_head=2,
                 n_embd=32, ffn_hidden=64)


def make(cfg=TINY):
    return GPT(cfg).eval()


def test_initial_loss_is_log_vocab():
    m = make()
    x = torch.randint(0, 64, (8, 16))
    y = torch.randint(0, 64, (8, 16))   # independent targets
    _, loss = m(x, y)
    assert abs(loss.item() - math.log(64)) < 0.3


def test_causality():
    m = make()
    a = torch.randint(0, 64, (1, 16))
    b = a.clone()
    b[0, 10] = (b[0, 10] + 1) % 64            # change one token in the middle
    la, _ = m(a)
    lb, _ = m(b)
    assert torch.allclose(la[:, :10], lb[:, :10], atol=1e-5)   # past unaffected
    assert not torch.allclose(la[:, 10:], lb[:, 10:], atol=1e-5)  # future affected


def test_rope_is_relative():
    cos, sin = build_rope(16, 64, 10000.0)
    q, k = torch.randn(1, 1, 1, 16), torch.randn(1, 1, 1, 16)

    def score(m, n):
        qm = apply_rope(q, cos[m:m + 1], sin[m:m + 1])
        kn = apply_rope(k, cos[n:n + 1], sin[n:n + 1])
        return (qm * kn).sum().item()

    assert abs(score(5, 2) - score(25, 22)) < 1e-4   # same offset -> same score


def test_kv_cache_matches_full_forward():
    m = make()
    ids = torch.randint(0, 64, (1, 20))
    full, _ = m(ids)
    _, kv = m.forward_cached(ids[:, :-1])
    inc, _ = m.forward_cached(ids[:, -1:], kv, ids.size(1) - 1)
    assert torch.allclose(full[:, -1], inc, atol=1e-4)


def test_greedy_cached_equals_nocache():
    m = make()
    prompt = torch.randint(0, 64, (1, 5))
    kw = dict(temperature=0.0, repetition_penalty=1.0)
    a = m.generate_nocache(prompt, 20, **kw)[0].tolist()
    b = prompt[0].tolist() + list(m.generate_stream(prompt, 20, **kw))
    assert a == b


def test_generates_past_context_window():
    m = make()
    prompt = torch.randint(0, 64, (1, 5))
    out = list(m.generate_stream(prompt, 100))   # block_size is 32
    assert len(out) == 100


def test_gqa_has_fewer_params_than_mha():
    mha = GPT(GPTConfig(**{**TINY.__dict__, "n_kv_head": 4}))
    gqa = GPT(TINY)
    assert gqa.num_params() < mha.num_params()


def test_top_p_and_top_k_sampling_runs():
    m = make()
    prompt = torch.randint(0, 64, (1, 5))
    out = list(m.generate_stream(prompt, 10, temperature=1.0, top_k=5, top_p=0.5))
    assert len(out) == 10 and all(0 <= t < 64 for t in out)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("PASS", name)
