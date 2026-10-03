"""
Train the model.

    python train.py                       # fresh run
    python train.py --resume              # continue from out/ckpt_last.pt
    caffeinate -i python train.py         # macOS: keep the Mac awake
"""
import argparse
import contextlib
import csv
import math
import os
import time
from dataclasses import asdict

import numpy as np
import torch

from inference import pick_device
from model import GPT, GPTConfig

p = argparse.ArgumentParser()
p.add_argument("--data_dir", default="data")
p.add_argument("--out_dir", default="out")
p.add_argument("--max_iters", type=int, default=6000)
p.add_argument("--batch_size", type=int, default=32)
p.add_argument("--grad_accum", type=int, default=1, help="micro-batches per optimizer step")
p.add_argument("--lr", type=float, default=1e-3)
p.add_argument("--min_lr_ratio", type=float, default=0.1)
p.add_argument("--warmup", type=int, default=200)
p.add_argument("--weight_decay", type=float, default=0.1)
p.add_argument("--grad_clip", type=float, default=1.0)
p.add_argument("--eval_interval", type=int, default=500)
p.add_argument("--eval_iters", type=int, default=50)
p.add_argument("--log_interval", type=int, default=50)
p.add_argument("--dtype", default="float32", choices=["float32", "bfloat16", "float16"],
               help="autocast dtype; float32 is the safe default on Apple GPUs")
p.add_argument("--resume", action="store_true")
p.add_argument("--seed", type=int, default=1337)
# architecture overrides
p.add_argument("--block_size", type=int, default=256)
p.add_argument("--n_layer", type=int, default=8)
p.add_argument("--n_head", type=int, default=6)
p.add_argument("--n_kv_head", type=int, default=2)
p.add_argument("--n_embd", type=int, default=384)
p.add_argument("--ffn_hidden", type=int, default=1024)
args = p.parse_args()

torch.manual_seed(args.seed)
device = pick_device()
os.makedirs(args.out_dir, exist_ok=True)

# ---- data ----
train_data = np.memmap(os.path.join(args.data_dir, "train.bin"), dtype=np.uint16, mode="r")
val_data = np.memmap(os.path.join(args.data_dir, "val.bin"), dtype=np.uint16, mode="r")
from tokenizers import Tokenizer
tok = Tokenizer.from_file(os.path.join(args.data_dir, "tokenizer.json"))
eot = tok.token_to_id("<|endoftext|>")


def get_batch(split, T):
    d = train_data if split == "train" else val_data
    ix = torch.randint(len(d) - T - 1, (args.batch_size,)).tolist()
    x = np.stack([d[i : i + T] for i in ix]).astype(np.int64)
    y = np.stack([d[i + 1 : i + T + 1] for i in ix]).astype(np.int64)
    return torch.from_numpy(x).to(device), torch.from_numpy(y).to(device)


# ---- model / optimizer ----
cfg = GPTConfig(vocab_size=tok.get_vocab_size(), block_size=args.block_size,
                n_layer=args.n_layer, n_head=args.n_head, n_kv_head=args.n_kv_head,
                n_embd=args.n_embd, ffn_hidden=args.ffn_hidden)
iter_num, best_val = 0, float("inf")
ckpt = None
if args.resume:
    ckpt = torch.load(os.path.join(args.out_dir, "ckpt_last.pt"), map_location=device)
    cfg = GPTConfig(**ckpt["config"])  # architecture comes from the checkpoint
model = GPT(cfg).to(device)
if ckpt:
    model.load_state_dict(ckpt["model"])
    iter_num, best_val = ckpt["iter"], ckpt["best_val"]

# weight decay on matrices only (not on norm gains)
decay = [p_ for p_ in model.parameters() if p_.dim() >= 2]
no_decay = [p_ for p_ in model.parameters() if p_.dim() < 2]
optimizer = torch.optim.AdamW(
    [{"params": decay, "weight_decay": args.weight_decay},
     {"params": no_decay, "weight_decay": 0.0}],
    lr=args.lr, betas=(0.9, 0.95))
if ckpt:
    optimizer.load_state_dict(ckpt["optimizer"])

print(f"device={device}  params={model.num_params()/1e6:.2f}M  config={asdict(cfg)}")
tokens_per_iter = args.batch_size * args.grad_accum * cfg.block_size
print(f"tokens/iter={tokens_per_iter:,}  total tokens this run="
      f"{(args.max_iters - iter_num) * tokens_per_iter / 1e6:.0f}M")

amp = (contextlib.nullcontext() if args.dtype == "float32"
       else torch.autocast(device_type=device, dtype=getattr(torch, args.dtype)))


def get_lr(it):
    if it < args.warmup:
        return args.lr * (it + 1) / args.warmup
    progress = min(1.0, (it - args.warmup) / max(1, args.max_iters - args.warmup))
    floor = args.lr * args.min_lr_ratio
    return floor + 0.5 * (args.lr - floor) * (1 + math.cos(math.pi * progress))


@torch.no_grad()
def estimate_loss():
    model.eval()
    out = {}
    for split in ["train", "val"]:
        losses = torch.zeros(args.eval_iters)
        for k in range(args.eval_iters):
            xb, yb = get_batch(split, cfg.block_size)
            with amp:
                _, loss = model(xb, yb)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def save(path, with_optimizer=True):
    state = {"model": model.state_dict(), "config": asdict(cfg), "iter": iter_num,
             "best_val": best_val}
    if with_optimizer:
        state["optimizer"] = optimizer.state_dict()
    torch.save(state, path)


def sample_preview():
    ids = torch.tensor([tok.encode("Once upon a time").ids], device=device)
    out = list(model.generate_stream(ids, 80, stop_id=eot))
    return tok.decode([t for t in out if t != eot])


log_path = os.path.join(args.out_dir, "log.csv")
if not os.path.exists(log_path):
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(["iter", "train_loss", "val_loss", "val_ppl", "lr", "elapsed_s", "tokens"])

# ---- training loop ----
model.train()
t_start = time.time()
t_log = time.time()
start_iter = iter_num
while iter_num < args.max_iters:
    lr_now = get_lr(iter_num)
    for g in optimizer.param_groups:
        g["lr"] = lr_now

    if iter_num % args.eval_interval == 0 and iter_num > start_iter or iter_num == args.max_iters - 1:
        losses = estimate_loss()
        ppl = math.exp(losses["val"])
        print(f"\n== step {iter_num}: train {losses['train']:.4f} | val {losses['val']:.4f} "
              f"| val ppl {ppl:.2f} | {time.time()-t_start:.0f}s")
        print("sample:", sample_preview().replace("\n", " ")[:300], "\n")
        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([iter_num, f"{losses['train']:.4f}", f"{losses['val']:.4f}",
                                    f"{ppl:.2f}", f"{lr_now:.2e}", f"{time.time()-t_start:.0f}",
                                    iter_num * tokens_per_iter])
        if losses["val"] < best_val:
            best_val = losses["val"]
            save(os.path.join(args.out_dir, "ckpt_best.pt"), with_optimizer=False)
        save(os.path.join(args.out_dir, "ckpt_last.pt"))

    loss_accum = 0.0
    for _ in range(args.grad_accum):
        xb, yb = get_batch("train", cfg.block_size)
        with amp:
            _, loss = model(xb, yb)
        (loss / args.grad_accum).backward()
        loss_accum += loss.item() / args.grad_accum
    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    iter_num += 1

    if iter_num % args.log_interval == 0:
        dt = time.time() - t_log
        t_log = time.time()
        done = iter_num - start_iter
        eta = (time.time() - t_start) / done * (args.max_iters - iter_num)
        print(f"iter {iter_num:5d} | loss {loss_accum:.4f} | lr {lr_now:.2e} | "
              f"{args.log_interval * tokens_per_iter / dt:,.0f} tok/s | eta {eta/60:.0f} min")

save(os.path.join(args.out_dir, "ckpt_last.pt"))
print(f"done. best val loss {best_val:.4f}. Weights: {args.out_dir}/ckpt_best.pt")
