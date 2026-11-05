"""exllamav3 reference benchmark, same protocol as tinycoder_bench:
   512-token prefill (pp512) and 128-token greedy decode (tg128), seq=1 batch.
"""
import sys
import time

import torch
from exllamav3 import Cache, Config, Model, Tokenizer, GreedySampler

MODEL = sys.argv[1] if len(sys.argv) > 1 else \
    "/data/models/qwen/qwen3.8-27b/qwen3.8-27B-exl3-1.60bpw"
PP, TG = 512, 128
MAXLEN = 1024

config = Config.from_directory(MODEL)
model = Model.from_config(config)
cache = Cache(model, max_num_tokens=MAXLEN)
model.load(progressbar=False, max_chunk_size=MAXLEN)
tok = Tokenizer.from_config(config)
sampler = GreedySampler()
print(f"loaded: {MODEL}", flush=True)

text = "The quick brown fox jumps over the lazy dog. " * 200
ids = tok.encode(text, encode_special_tokens=True)[:PP]
ids = ids[:, :PP].to("cuda")
assert ids.shape[-1] == PP, ids.shape

def params(past):
    return {
        "attn_mode": "flash_attn",
        "cache": cache,
        "past_len": past,
        "batch_shape": (1, MAXLEN),
    }

# Single timed prefill per process (the cache has no reset API; the callers run
# this script three times, matching the 3-run protocol used for tinycoder).
p = params(0)
torch.cuda.synchronize()
t0 = time.time()
model.prefill(input_ids=ids, params=p)
torch.cuda.synchronize()
pp_s = time.time() - t0

rs = p.get("recurrent_states")
nxt = ids[:, -1:]
past = PP
out_ids = []
torch.cuda.synchronize()
t0 = time.time()
for _ in range(TG):
    q = params(past)
    if rs is not None:
        q["recurrent_states"] = rs
    logits = model.forward(input_ids=nxt, params=q)
    nxt = sampler.forward(logits, tokenizer=tok).reshape(1, 1)
    out_ids.append(int(nxt[0, 0]))
    past += 1
torch.cuda.synchronize()
tg_s = time.time() - t0

print("SAMPLE: " + repr(tok.decode(torch.tensor(out_ids))) )
print(f"RESULT pp{PP} | {PP / pp_s:8.1f} tok/s ({pp_s * 1000:7.1f} ms)")
print(f"RESULT tg{TG} | {TG / tg_s:8.2f} tok/s ({tg_s * 1000:7.1f} ms)")
