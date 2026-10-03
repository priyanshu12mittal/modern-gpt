"""Loads a checkpoint + tokenizer and streams text. Shared by generate.py and serve.py."""
import torch
from tokenizers import Tokenizer

from model import GPT


def pick_device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class Engine:
    def __init__(self, ckpt="out/ckpt_best.pt", tokenizer="data/tokenizer.json", device=None):
        self.device = device or pick_device()
        self.model, self.ckpt = GPT.from_checkpoint(ckpt, self.device)
        self.tok = Tokenizer.from_file(tokenizer)
        self.eot = self.tok.token_to_id("<|endoftext|>")

    def stream(self, prompt, max_new_tokens=200, **sampling):
        """Yields text chunks as tokens are generated."""
        ids = self.tok.encode(prompt).ids or [self.eot]
        idx = torch.tensor([ids], device=self.device)
        out_ids, emitted = [], ""
        for tid in self.model.generate_stream(idx, max_new_tokens, stop_id=self.eot, **sampling):
            if tid == self.eot:
                break
            out_ids.append(tid)
            text = self.tok.decode(out_ids)
            if text.endswith("\ufffd"):  # incomplete UTF-8 sequence: wait for more tokens
                continue
            delta = text[len(emitted):]
            emitted = text
            if delta:
                yield delta
        tail = self.tok.decode(out_ids)[len(emitted):]
        if tail:
            yield tail

    def complete(self, prompt, max_new_tokens=200, **sampling):
        return "".join(self.stream(prompt, max_new_tokens, **sampling))
