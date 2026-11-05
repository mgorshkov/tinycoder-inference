"""Per-layer sub-branch magnitudes from exllamav3 (attention and MLP outputs).

Where `exl3_layer_state_norms.py` gives the block output, this separates the two
contributions: y_attn and y_mlp are the tensors added into the residual stream.

Compare against the engine's `[gpuM L]` / `[gpuL L] attnProjNorm=` dumps:

    TINYCODER_DUMP_LAYERS=1 TINYCODER_AGREE_SWEEP=1 TINYCODER_AGREE_LEN=<N> \
      ./build-1.0/unit_tests/exl3_decode_agree --model <ckpt> --max-pos <N>
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

blocks = [m for m in model.modules if type(m).__name__ == "TransformerBlock"]
attn_norm, mlp_norm = {}, {}


def wrap(mod, store, i):
    """exllamav3 modules are not nn.Modules, so patch the bound forward."""
    orig = mod.forward

    def forward(x, params, *a, **kw):
        y = orig(x, params, *a, **kw)
        t = y[0] if isinstance(y, (tuple, list)) else y
        if torch.is_tensor(t) and t.dim() == 3:
            store[i] = float(t[0, -1].float().norm())
        return y

    mod.forward = forward


emb_norm = {}
for m in model.modules:
    if type(m).__name__ == "Embedding":
        wrap(m, emb_norm, 0)

for i, b in enumerate(blocks):
    wrap(b.attn, attn_norm, i)
    wrap(b.mlp, mlp_norm, i)

p = {"attn_mode": "flash_attn", "cache": cache, "past_len": 0,
     "batch_shape": (1, MAXLEN)}
model.forward(input_ids=torch.tensor([IDS[:N]], dtype=torch.long), params=p)

print(f"ORACLE_EMB norm={emb_norm.get(0, float('nan')):10.4f}")
for i in range(len(blocks)):
    a = attn_norm.get(i, float("nan"))
    m = mlp_norm.get(i, float("nan"))
    kind = "attn" if (i + 1) % 4 == 0 else "gdn"
    print(f"ORACLE_BRANCH L{i:3d} {kind:4s} attn={a:10.4f} mlp={m:10.4f}")
