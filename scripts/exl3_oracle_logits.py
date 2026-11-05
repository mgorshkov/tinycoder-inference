"""Oracle logits for an explicit token-id list, from exllamav3's own forward.

The point is to compare TinyCoder's trunk against a known-good engine on the
*same ids*, so a tokenizer or chat-template difference cannot hide a math bug:

    python scripts/exl3_oracle_logits.py <checkpoint> 248045 846 198 ...

Prints the last position's top-5 (id, logit) in the same format the
exl3_decode_agree sweep prints, plus the initial-state logits of the first
position for a shorter cross-check.
"""
import sys

import torch
from exllamav3 import Cache, Config, Model, Tokenizer

MODEL = sys.argv[1]
RAW = sys.argv[2:]
# --decode-step: also compute the logits for the LAST id via a seqLen==1
# forward on top of a prefill of ids[:-1], i.e. the stateful decode path, and
# print both.  A difference here is the decode path's own error (bf16 KV cache
# rounding, recurrent-state update) rather than a prefill math bug.
DECODE_STEP = "--decode-step" in RAW
IDS = [int(x) for x in RAW if not x.startswith("--")]
MAXLEN = 256

config = Config.from_directory(MODEL)
model = Model.from_config(config)
cache = Cache(model, max_num_tokens=MAXLEN)
model.load(progressbar=False, max_chunk_size=MAXLEN)
tok = Tokenizer.from_config(config)


def params(past):
    return {"attn_mode": "flash_attn", "cache": cache, "past_len": past,
            "batch_shape": (1, MAXLEN)}


ids = torch.tensor([IDS], dtype=torch.long)
p0 = params(0)
logits = model.forward(input_ids=ids, params=p0)
row = logits[0, -1].float()
top = torch.topk(row, 5)
print("ORACLE top5=" + ",".join(
    f"{int(i)}:{float(v):.4f}" for v, i in zip(top.values, top.indices)))
print("ORACLE top1 text=", repr(tok.decode(torch.tensor([int(top.indices[0])]))))
if DECODE_STEP:
    pa = params(0)
    model.forward(input_ids=ids[:, :-1], params=pa)
    rs = pa.get("recurrent_states")
    pb = params(IDS.__len__() - 1)
    if rs is not None:
        pb["recurrent_states"] = rs
    lg = model.forward(input_ids=ids[:, -1:], params=pb)
    row2 = lg[0, -1].float()
    top2 = torch.topk(row2, 5)
    print("ORACLE decode top5=" + ",".join(
        f"{int(i)}:{float(v):.4f}" for v, i in zip(top2.values, top2.indices)))
    d = (row2 - row).abs().max().item()
    print(f"ORACLE prefill-vs-decode maxAbsDiff={d:.6g}")
    sys.exit(0)
# Greedy continuation with the recurrent/KV state carried, mirroring the
# engine's own decode loop (so it also exercises the delta-net state).
# Greedy continuation.  The prefill above already produced `logits` for the
# last position AND filled the KV cache / recurrent state, so the first
# generated token must be argmax of THOSE logits.  Feeding ids[:, -1:] again
# samples from a context with the final prompt token duplicated (an off-by-one
# that made every "oracle continuation" answer a slightly different prompt).
rs = p0.get("recurrent_states")           # set by the forward() above
past = ids.shape[-1]
gen = []
for _ in range(12):
    nxt = torch.argmax(logits[0, -1]).reshape(1, 1)
    gen.append(int(nxt[0, 0]))
    q = params(past)
    if rs is not None:
        q["recurrent_states"] = rs
    logits = model.forward(input_ids=nxt, params=q)
    past += 1
print("ORACLE greedy continuation ids:", gen)
print("ORACLE greedy continuation text:", repr(tok.decode(torch.tensor(gen))))
