"""Stage-by-stage GDN census from exllamav3, to bisect a gated-delta-net branch.

The engine's `TINYCODER_DUMP_GDN=1` prints one line per recurrent layer with the
norm of every stage (projected qkv, conv output, delta-net output, gate z,
gated-norm output, projected branch).  This prints the same quantities, on the
same row (LAST token), from exllamav3 on the same ids, so a branch that is 3.6 %
small can be attributed to a stage instead of guessed at:

    PATH=/tmp/exl3venv/bin:$PATH /tmp/exl3venv/bin/python \
      scripts/exl3_gdn_stage_probe.py <ckpt> <N> <ids...> [--layers 0,1,4]

Stages are recorded by patching the module-level conv / delta-rule entry points
and the GatedRMSNorm forward; the GDN call order (one conv per recurrent layer,
in stack order) names the layer.
"""
import sys

import torch
from exllamav3 import Cache, Config, Model
import exllamav3.modules.gated_delta_net as gdnmod
from exllamav3.modules.gated_rmsnorm import GatedRMSNorm

MODEL = sys.argv[1]
N = int(sys.argv[2])
args = sys.argv[3:]
want_layers = None
if "--layers" in args:
    j = args.index("--layers")
    want_layers = {int(x) for x in args[j + 1].split(",")}
    args = args[:j] + args[j + 2:]
IDS = [int(x) for x in args]
MAXLEN = 256

config = Config.from_directory(MODEL)
model = Model.from_config(config)
cache = Cache(model, max_num_tokens=MAXLEN)
model.load(progressbar=False, max_chunk_size=MAXLEN)


def last_row(x, seqlen):
    """Row of the LAST token, from either layout.

    (bsz, seqlen, ...) and (bsz, seqlen, heads, dim) keep the sequence axis at 1;
    a channel-major (bsz, dim, seqlen) keeps it at 2.
    """
    if not torch.is_tensor(x):
        return None
    x = x.detach()
    if x.dim() == 1:
        return x.float()
    if x.dim() == 2:
        return x[-1].float()
    shp = x.shape
    if shp[1] == seqlen:
        r = x[0, -1]
    elif shp[2] == seqlen:
        r = x[0, :, -1]
    else:
        r = x[0, -1]
    return r.float().reshape(-1)


def stats(x, seqlen):
    r = last_row(x, seqlen)
    if r is None:
        return (float("nan"), [])
    return (float(r.norm()), [float(v) for v in r[:6]])


conv_in, conv_out, delta_out, gate_in, norm_out = {}, {}, {}, {}, {}
cur = {"i": -1}
seqlen = {"n": 0}

_orig_conv = gdnmod.causal_conv1d_update
_orig_delta = gdnmod.gated_delta_rule_fn
_orig_norm = GatedRMSNorm.forward


def conv_hook(*a, **kw):
    cur["i"] += 1
    i = cur["i"]
    conv_in[i] = stats(kw.get("mixed_qkv"), seqlen["n"])
    out = _orig_conv(*a, **kw)
    conv_out[i] = stats(out, seqlen["n"])
    return out


def delta_hook(*a, **kw):
    i = cur["i"]
    out = _orig_delta(*a, **kw)
    delta_out[i] = stats(out, seqlen["n"])
    g = kw.get("g")
    b = kw.get("beta")
    gt = last_row(g, seqlen["n"]) if g is not None else None
    bt = last_row(b, seqlen["n"]) if b is not None else None
    delta_out[i] = delta_out[i] + (
        "",
        float(bt.norm()) if bt is not None else float("nan"),
        "" if gt is None else f"[{float(gt.min()):+.4f},{float(gt.max()):+.4f}]",
    )
    return out


def norm_hook(self, x, params, *a, **kw):
    i = cur["i"]
    gate_in[i] = stats(kw.get("gate"), seqlen["n"])
    y = _orig_norm(self, x, params, *a, **kw)
    norm_out[i] = stats(y, seqlen["n"])
    return y


gdnmod.causal_conv1d_update = conv_hook
gdnmod.gated_delta_rule_fn = delta_hook
GatedRMSNorm.forward = norm_hook

oproj_out = {}
for m in model.modules:
    if type(m).__name__ == "TransformerBlock" and type(m.attn).__name__ == "GatedDeltaNet":
        op = m.attn.o_proj
        orig = op.forward
        idx = m.attn.layer_idx

        def make(orig=orig, idx=idx):
            def forward(x, params, *a, **kw):
                y = orig(x, params, *a, **kw)
                oproj_out[idx] = stats(y, seqlen["n"])
                return y

            return forward

        op.forward = make()

seqlen["n"] = N
ids = torch.tensor([IDS[:N]], dtype=torch.long)
p = {"attn_mode": "flash_attn", "cache": cache, "past_len": 0,
     "batch_shape": (1, MAXLEN)}
model.forward(input_ids=ids, params=p)

print(f"# n={N} gdn layers captured: {len(delta_out)} (last token, row-norms)")
for i in sorted(delta_out):
    if want_layers is not None and i not in want_layers:
        continue
    ci, civ = conv_in.get(i, (float("nan"), []))
    co, cov = conv_out.get(i, (float("nan"), []))
    do, dov, _, bn, grng = delta_out.get(i, (float("nan"), [], "", float("nan"), ""))
    gi, giv = gate_in.get(i, (float("nan"), []))
    no, nov = norm_out.get(i, (float("nan"), []))
    oo, oov = oproj_out.get(i, (float("nan"), []))

    def f6(v):
        return "[" + " ".join(f"{x:+.5f}" for x in v[:6]) + "]"

    print(f"ORACLE_GDN L{i:3d} qkvProj={ci:9.4f} qkv6={f6(civ)}")
    print(f"           convOut={co:9.4f} conv6={f6(cov)}")
    print(f"           deltaOut={do:9.4f} betaN={bn:7.4f} gate6={f6(giv)} "
          f"gateRange={grng}")
    print(f"           delta6={f6(dov)}")
    print(f"           gatedNormOut={no:9.4f} gn6={f6(nov)} oProjOut={oo:9.4f} "
          f"o6={f6(oov)}")
