"""Download TinyStories, train a byte-level BPE tokenizer, write train.bin / val.bin.

    python prepare_data.py --n_stories 100000

Already have tokenizer.json / train.bin / val.bin from the earlier mini-gpt project?
Just copy them into ./data and skip this script.
"""
import argparse
import os

import numpy as np
from datasets import load_dataset
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

p = argparse.ArgumentParser()
p.add_argument("--n_stories", type=int, default=100_000)
p.add_argument("--vocab_size", type=int, default=4096)
p.add_argument("--out_dir", default="data")
args = p.parse_args()
os.makedirs(args.out_dir, exist_ok=True)

ds = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
stories = []
for i, ex in enumerate(ds):
    if i >= args.n_stories:
        break
    stories.append(ex["text"].strip())
print(f"downloaded {len(stories):,} stories")

tok = Tokenizer(models.BPE())
tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
tok.decoder = decoders.ByteLevel()
trainer = trainers.BpeTrainer(
    vocab_size=args.vocab_size,
    special_tokens=["<|endoftext|>"],
    initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
)
tok.train_from_iterator(stories, trainer)
tok.save(os.path.join(args.out_dir, "tokenizer.json"))

eot = tok.token_to_id("<|endoftext|>")
ids = []
for enc in tok.encode_batch(stories):
    ids.extend(enc.ids + [eot])
ids = np.array(ids, dtype=np.uint16)

n = int(0.95 * len(ids))
ids[:n].tofile(os.path.join(args.out_dir, "train.bin"))
ids[n:].tofile(os.path.join(args.out_dir, "val.bin"))
print(f"total tokens: {len(ids):,}  (train {n:,}, val {len(ids) - n:,})")
