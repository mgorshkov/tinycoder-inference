"""Per-layer residual-stream fingerprint from exllamav3, one line per layer.

Norms alone hid where the two engines diverge (a norm can agree while the
direction is wrong, and a 0.73 vs 0.82 ratio band is not enough to tell a
constant deficit from a multiplicative one).  This prints the same
`first8` components the engine's `[gpuL n] post-ffn` dump prints, plus the
per-branch (attn / mlp) norms, so a divergence can be attributed to a layer
AND to a branch in one run:

    PATH=/tmp/exl3venv/bin:$PATH /tmp/exl3venv/bin/python \
      scripts/exl3_layer_residual_probe.py <ckpt> <N> <ids...>

Compare against:

    TINYCODER_DUMP_LAYERS=1 TINYCODER_AGREE_SWEEP=1 TINYCODER_AGREE_LEN=<N> \
      ./build-1.0/unit_tests/exl3_decode_agree --model <ckpt> --max-pos <N>

`export_state_layers` disables exllamav3's fused deferred residual for the
exported layers, so the exported tensor is the real post-FFN residual stream
(the same point the engine dumps after its `kAddResidual(ffnOut)`).
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

blocks = [m for m in model.modules if type(m).__name__ == "TransformerBlock"]
for i, b in enumerate(blocks):
    wrap(b.attn, attn_norm, i)
    wrap(b.mlp, mlp_norm, i)

ids = torch.tensor([IDS[:N]], dtype=torch.long)
p = {"attn_mode": "flash_attn", "cache": cache, "past_len": 0,
     "batch_shape": (1, MAXLEN),
     "export_state_layers": set(range(config.num_hidden_layers))}
model.forward(input_ids=ids, params=p)
states = p["export_states"]
print(f"# exported {len(states)} layers, shape {tuple(states[0].shape)}, "
      f"n={N} emb_norm={emb_norm.get(0, float('nan')):.6f}")
for i, st in enumerate(states):
    row = st[0, -1].float()
    bad = int((~torch.isfinite(row)).sum())
    f8 = " ".join(f"{float(v):.6f}" for v in row[:8])
    kind = "attn" if (i + 1) % 4 == 0 else "gdn"
    print(f"ORACLE_STATE L{i:3d} {kind:4s} norm={float(row.norm()):10.4f} "
          f"maxabs={float(row.abs().max()):10.4f} nonfinite={bad} "
          f"attn={attn_norm.get(i, float('nan')):9.4f} "
          f"mlp={mlp_norm.get(i, float('nan')):9.4f} first8=[{f8}]")
