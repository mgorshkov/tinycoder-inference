"""Oracle top-5 logits from exllamav3 for a prefix of an explicit token-id list.

One process per length: exllamav3 attaches its KV pages and recurrent-state slots
at load time, and a fresh Cache per measurement is the only reliable way to
guarantee a clean delta-net state.

    python scripts/exl3_oracle_prefix.py <checkpoint> <N> <ids...>

Compare against TinyCoder:

    TINYCODER_AGREE_SWEEP=1 TINYCODER_AGREE_LEN=<N> \
      ./build-1.0/unit_tests/exl3_decode_agree --model <checkpoint> --max-pos <N>
"""
import sys

import torch
from exllamav3 import Cache, Config, Model

MODEL = sys.argv[1]
N = int(sys.argv[2])
IDS = [int(x) for x in sys.argv[3:]]
MAXLEN = 256

config = Config.from_directory(MODEL)
model = Model.from_config(config)
cache = Cache(model, max_num_tokens=MAXLEN)
model.load(progressbar=False, max_chunk_size=MAXLEN)

ids = torch.tensor([IDS[:N]], dtype=torch.long)
p = {"attn_mode": "flash_attn", "cache": cache, "past_len": 0,
     "batch_shape": (1, MAXLEN)}
logits = model.forward(input_ids=ids, params=p)
row = logits[0, -1].float()
top = torch.topk(row, 5)
print(f"ORACLE n={N:3d} finite={bool(torch.isfinite(row).all())} "
      f"rms={float(row.pow(2).mean().sqrt()):.4f} "
      f"top5=" + ",".join(f"{int(i)}:{float(v):.4f}"
                          for v, i in zip(top.values, top.indices)))
