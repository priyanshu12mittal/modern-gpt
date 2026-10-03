"""
Generate stories from the command line.

    python generate.py --prompt "Once upon a time" --n 3
    python generate.py --bench            # KV-cache speed benchmark
"""
import argparse
import time

import torch

from inference import Engine

p = argparse.ArgumentParser()
p.add_argument("--ckpt", default="out/ckpt_best.pt")
p.add_argument("--tokenizer", default="data/tokenizer.json")
p.add_argument("--prompt", default="Once upon a time")
p.add_argument("--max_new_tokens", type=int, default=250)
p.add_argument("--temperature", type=float, default=0.8)
p.add_argument("--top_k", type=int, default=40)
p.add_argument("--top_p", type=float, default=0.95)
p.add_argument("--repetition_penalty", type=float, default=1.1)
p.add_argument("--n", type=int, default=1, help="number of samples")
p.add_argument("--seed", type=int, default=None)
p.add_argument("--bench", action="store_true")
args = p.parse_args()

eng = Engine(args.ckpt, args.tokenizer)
if args.seed is not None:
    torch.manual_seed(args.seed)
sampling = dict(temperature=args.temperature, top_k=args.top_k, top_p=args.top_p,
                repetition_penalty=args.repetition_penalty)

if args.bench:
    dev, N = eng.device, 200
    ids = torch.tensor([eng.tok.encode(args.prompt).ids], device=dev)
    sync = torch.mps.synchronize if dev == "mps" else (lambda: None)
    list(eng.model.generate_stream(ids, 5)); eng.model.generate_nocache(ids, 5)  # warm-up
    sync(); t = time.time(); eng.model.generate_nocache(ids, N); sync(); t_plain = time.time() - t
    sync(); t = time.time(); list(eng.model.generate_stream(ids, N)); sync(); t_cache = time.time() - t
    print(f"no cache: {N / t_plain:6.0f} tok/s\nkv cache: {N / t_cache:6.0f} tok/s  "
          f"({t_plain / t_cache:.1f}x faster)")
else:
    for i in range(args.n):
        if args.n > 1:
            print(f"--- sample {i + 1} ---")
        print(args.prompt, end="", flush=True)
        for chunk in eng.stream(args.prompt, args.max_new_tokens, **sampling):
            print(chunk, end="", flush=True)
        print("\n")
