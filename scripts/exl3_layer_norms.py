"""Dump per-layer residual-stream norms from exllamav3 (known-good reference).

Used to answer two questions about the TinyCoder qwen35 forward:
  1. Does the residual stream norm legitimately GROW with depth in this model?
  2. What does a correct answer to a factual question look like (top-5 logits)?
"""
import sys

import torch
from exllamav3 import Cache, Config, Model, Tokenizer, GreedySampler

MODEL = sys.argv[1] if len(sys.argv) > 1 else \
    "/data/models/qwen/qwen3.8-27b/qwen3.8-27B-exl3-1.60bpw"
QUESTION = sys.argv[2] if len(sys.argv) > 2 else "What is the capital of France?"
MAXLEN, N = 1024, 24

config = Config.from_directory(MODEL)
model = Model.from_config(config)
cache = Cache(model, max_num_tokens=MAXLEN)
model.load(progressbar=False, max_chunk_size=MAXLEN)
tok = Tokenizer.from_config(config)
sampler = GreedySampler()

prompt = ("<|im_start|>user\n" + QUESTION + "<|im_end|>\n<|im_start|>assistant\n")
ids = tok.encode(prompt, encode_special_tokens=True)
print(f"prompt tokens: {ids.shape[-1]}")

mm = model
layer_mods = [m for m in mm.modules if type(m).__name__ == "TransformerBlock"]
print(f"transformer blocks: {len(layer_mods)}")

norms = {}


def mk_hook(i):
    def hook(module, inp, out):
        # forward returns the residual stream (bsz, seq, hidden) or a tuple
        x = out[0] if isinstance(out, (tuple, list)) else out
        if not torch.is_tensor(x):
            return
        last = x[0, -1].float() if x.dim() == 3 else x.float()
        norms[i] = (float(last.norm()), torch.isfinite(last).all().item())
    return hook


for i, m in enumerate(layer_mods):
    m.register_forward_hook(mk_hook(i))


def params(past):
    return {"attn_mode": "flash_attn", "cache": cache, "past_len": past,
            "batch_shape": (1, MAXLEN)}


p = params(0)
model.prefill(input_ids=ids, params=p)
rs = p.get("recurrent_states")

print("=== exllamav3 per-layer post-FFN residual norm (last token, prefill) ===")
for i in sorted(norms):
    n, fin = norms[i]
    print(f"  L{i:3d} norm={n:10.4f} finite={fin}")

nxt = ids[:, -1:]
past = ids.shape[-1]
out = []
for _ in range(N):
    q = params(past)
    if rs is not None:
        q["recurrent_states"] = rs
    logits = model.forward(input_ids=nxt, params=q)
    nxt = sampler.forward(logits, tokenizer=tok).reshape(1, 1)
    out.append(int(nxt[0, 0]))
    past += 1

txt = tok.decode(torch.tensor(out))
print("ANSWER:", repr(txt))
