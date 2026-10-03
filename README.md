# Mini GPT

A **~14M-parameter decoder-only transformer**, written in plain PyTorch (no `transformers` library),
trained on [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) on a MacBook Air (M4 / MPS).

Includes a byte-level BPE tokenizer, a full training loop with cosine LR decay,
KV-cache inference, a streaming HTTP API, and a web UI — all from scratch.

## Architecture

| Component | Choice | Why |
|---|---|---|
| Positions | **RoPE** (rotary embeddings) | Relative positions, no learned position table |
| Normalisation | **RMSNorm**, pre-norm | Cheaper and as stable as LayerNorm |
| Feed-forward | **SwiGLU** | Better quality per parameter than GELU MLP |
| Attention | **Grouped-query attention** (6 query heads, 2 KV heads) | KV cache is 3× smaller |
| Embeddings | Tied input / output | Saves parameters |
| Init | N(0, 0.02); residual projections scaled by 1/√(2·layers) | Stable training at depth |
| Biases | None | Standard in modern LLMs |
| Tokenizer | Byte-level BPE, 4 096 vocab, trained from scratch | No unknown tokens |

**Training:** AdamW (decoupled weight decay on matrices only), linear warmup + cosine decay, gradient clipping,
gradient accumulation, checkpoint / resume, CSV logging.

**Inference:** KV cache, temperature, top-k, top-p (nucleus) sampling, repetition penalty, streaming.

## Results

| Metric | Value |
|---|---|
| Parameters | 14.2 M |
| Training data | ~49 M tokens of TinyStories (100k stories) |
| Training steps / time | 6 000 steps / ~160 min on MacBook Air M4 |
| Best val loss / perplexity | **1.70** / **5.48** |
| Generation speed (no cache → KV cache) | 69 → **194 tok/s** (2.8× speedup) |

### Training curve

| Step | Train loss | Val loss | Val PPL |
|-----:|-----------:|---------:|--------:|
|  500 |     2.6626 |   2.6477 |   14.12 |
| 1000 |     2.3107 |   2.2766 |    9.74 |
| 2000 |     2.0328 |   2.0045 |    7.42 |
| 3000 |     1.8742 |   1.8880 |    6.61 |
| 4000 |     1.7564 |   1.8059 |    6.09 |
| 5000 |     1.6782 |   1.7367 |    5.68 |
| 5999 |     1.6355 |   1.7005 |    5.48 |

Full log: [`out/log.csv`](out/log.csv)

### Sample outputs

> **Prompt:** "Once upon a time"

**Sample 1** — The model produces coherent multi-paragraph stories with dialogue:

> Once upon a time, there was a little girl named Lily. She loved to play with her toys and
> drink water from the cupboard. One day, she went to the kitchen to smell some juice. She
> tried to eat it, but it was too heavy.
>
> Lily asked her mom to help her pick up the cup. Her mom said no because they couldn't go to the …

**Sample 2** — It handles character names, emotions, and simple plot arcs:

> Once upon a time, there was a boy named Timmy. Timmy loved to play with his toys and eat
> fruit. One day, Timmy's mom told him that it's okay to have a feast so he stays up late and
> needs some help instead. Timmy got very sick and couldn't play with his favorite toys anymore.

Stories are grammatical and locally coherent but can drift or contradict themselves over longer spans — expected for a 14M-param model.

## Quick start

```bash
# create a virtual environment
python3 -m venv llm && source llm/bin/activate
pip install -r requirements.txt

# run the tests (no GPU / no data needed)
python test_model.py

# download TinyStories, train tokenizer, write data/*.bin
python prepare_data.py

# train (~2.5 hrs on M4 MacBook Air; Ctrl+C is safe, use --resume to continue)
caffeinate -i python train.py

# generate stories
python generate.py --prompt "Once upon a time" --n 3

# benchmark KV-cache speedup
python generate.py --bench

# start the API + web UI
uvicorn serve:app --port 8000    # → http://localhost:8000
```

> Already have `tokenizer.json`, `train.bin`, `val.bin` from an earlier run?
> Copy them into `data/` and skip `prepare_data.py`.

### API

```bash
# streaming (plain text, token by token)
curl -N -X POST localhost:8000/generate \
  -H 'Content-Type: application/json' \
  -d '{"prompt": "The little robot", "temperature": 0.8, "max_new_tokens": 150}'

# non-streaming (JSON response)
curl -X POST localhost:8000/complete \
  -H 'Content-Type: application/json' \
  -d '{"prompt": "The little robot", "max_new_tokens": 150}'
```

| Endpoint | Method | Returns |
|---|---|---|
| `/` | GET | Web UI |
| `/info` | GET | Model size, config, training stats (JSON) |
| `/generate` | POST | Streaming plain text |
| `/complete` | POST | JSON `{"text": "...", "seconds": ...}` |

## Project structure

```
├── model.py          # GPT model: RoPE, RMSNorm, SwiGLU, GQA, KV cache, sampling
├── train.py          # training loop with warmup, cosine decay, eval, checkpointing
├── inference.py      # Engine class: loads checkpoint + tokenizer, streams text
├── generate.py       # CLI: generate stories or benchmark KV cache
├── serve.py          # FastAPI server with streaming API + web UI
├── prepare_data.py   # download TinyStories, train BPE tokenizer, write binary shards
├── test_model.py     # 8 correctness tests (causality, RoPE, KV cache, GQA, …)
├── requirements.txt
├── data/
│   ├── tokenizer.json
│   ├── train.bin
│   └── val.bin
└── out/
    ├── ckpt_best.pt  # best checkpoint (val loss)
    ├── ckpt_last.pt  # last checkpoint (with optimizer state, for --resume)
    └── log.csv       # training metrics
```

## How the tests prove correctness

| Test | What it checks |
|---|---|
| `test_initial_loss_is_log_vocab` | Untrained loss ≈ ln(vocab_size) — embeddings are properly random |
| `test_causality` | Changing token *t* never changes logits at positions before *t* |
| `test_rope_is_relative` | Attention scores depend only on relative offset |
| `test_kv_cache_matches_full_forward` | Cached logits equal full forward pass (max diff ~1e-5) |
| `test_greedy_cached_equals_nocache` | Greedy decoding with/without KV cache gives identical tokens |
| `test_generates_past_context_window` | Generating past the 32-token context window works (cache rebuild) |
| `test_gqa_has_fewer_params_than_mha` | GQA model has fewer params than equivalent MHA model |
| `test_top_p_and_top_k_sampling_runs` | Nucleus + top-k sampling runs without errors |

## Limitations

- Small model, small data: stories are grammatical but drift, repeat, and contradict themselves.
- Context is 256 tokens. When it fills, the cache is rebuilt from the most recent 128 tokens.
- Trained only on simple children's stories; it cannot follow instructions.
- Single-user server (one generation at a time).

## Ideas to extend

- Train on more data / longer context
- Sliding-window attention
- Instruction fine-tuning
- int8 quantisation
- Compare against fine-tuned GPT-2 / Qwen-0.5B

## License

MIT
