"""Per-layer residual-stream norms from exllamav3's own forward (export_states).

The engine's `[gpuL n] post-ffn norm=` dump prints the same quantity, so this is
a direct oracle for it:

    TINYCODER_DUMP_LAYERS=1 TINYCODER_AGREE_SWEEP=1 TINYCODER_AGREE_LEN=37 \
      ./build-1.0/unit_tests/exl3_decode_agree --model <ckpt> --max-pos 37

    python scripts/exl3_layer_state_norms.py <ckpt> <N> <ids...>

`export_state_layers` makes the block fully add its residual (the fused
resid_defer path is disabled for exported layers), so the exported tensor is the
real post-FFN residual stream.
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
     "batch_shape": (1, MAXLEN), "export_state_layers": set(range(config.num_hidden_layers))}
model.forward(input_ids=ids, params=p)
states = p["export_states"]
print(f"exported {len(states)} layers, shape {tuple(states[0].shape)}")
for i, st in enumerate(states):
    row = st[0, -1].float()
    bad = int((~torch.isfinite(row)).sum())
    print(f"ORACLE_STATE L{i:3d} norm={float(row.norm()):10.4f} "
          f"maxabs={float(row.abs().max()):10.4f} nonfinite={bad}")
