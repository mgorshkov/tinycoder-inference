"""Does exllamav3 actually *infer* on this checkpoint? Ask a question, print the answer.

    python scripts/exl3_ask.py <ckpt> "question"
    python scripts/exl3_ask.py <ckpt> --prompt-file <path> [--tokens N]
                                       [--temp T] [--topp P] [--topk K]

--prompt-file feeds the exact bytes an engine produced (e.g. the `formatChat
prompt escaped:` line from tiny_logits_probe --generate) so a chat-template
difference cannot hide a math difference behind a different token stream.
"""
import sys

import torch
from exllamav3 import Cache, Config, Model, Tokenizer, GreedySampler

MODEL = sys.argv[1] if len(sys.argv) > 1 else \
    "/data/models/qwen/qwen3.8-27b/qwen3.8-27B-exl3-1.60bpw"
ARGV = sys.argv[2:]
PROMPT_FILE = None
MAXLEN, N = 1024, 48
TEMP, TOPP, TOPK = 0.0, 1.0, 0
i = 0
QUESTION = "What is the capital of France?"
while i < len(ARGV):
    a = ARGV[i]
    if a == "--prompt-file":
        PROMPT_FILE = ARGV[i + 1]; i += 2
    elif a == "--tokens":
        N = int(ARGV[i + 1]); i += 2
    elif a == "--temp":
        TEMP = float(ARGV[i + 1]); i += 2
    elif a == "--topp":
        TOPP = float(ARGV[i + 1]); i += 2
    elif a == "--topk":
        TOPK = int(ARGV[i + 1]); i += 2
    else:
        QUESTION = a; i += 1

config = Config.from_directory(MODEL)
model = Model.from_config(config)
cache = Cache(model, max_num_tokens=MAXLEN)
model.load(progressbar=False, max_chunk_size=MAXLEN)
tok = Tokenizer.from_config(config)
if TEMP > 0.0:
    try:
        from exllamav3 import Sampler, SamplerSettings
        sampler = Sampler(SamplerSettings(temperature=TEMP, top_p=TOPP, top_k=TOPK))
    except Exception as exc:      # keep the script usable across exl3 versions
        print(f"# sampler unavailable ({exc}); using greedy", file=sys.stderr)
        sampler = GreedySampler()
else:
    sampler = GreedySampler()

# Qwen/ChatML prompt, the same shape the TinyCoder question tests use.
if PROMPT_FILE is not None:
    prompt = open(PROMPT_FILE, "rb").read().decode("utf-8")
else:
    prompt = ("<|im_start|>user\n" + QUESTION + "<|im_end|>\n<|im_start|>assistant\n")
ids = tok.encode(prompt, encode_special_tokens=True)

def params(past):
    return {"attn_mode": "flash_attn", "cache": cache, "past_len": past,
            "batch_shape": (1, MAXLEN)}

# Standard sampling loop: the FIRST token comes from the prefill's logits.
# (Re-feeding the last prompt token -- forward(ids[:, -1:]) after prefill --
# samples from a context whose final token appears twice, which silently answers
# a different prompt: for the C++ question it turns "assistant\n" into
# "assistant\n\n" and the model then opens a code block instead of thinking.)
p = params(0)
logits = model.forward(input_ids=ids, params=p)
rs = p.get("recurrent_states")

past = ids.shape[-1]
out = []
for _ in range(N):
    nxt = sampler.forward(logits, tokenizer=tok).reshape(1, 1)
    out.append(int(nxt[0, 0]))
    q = params(past)
    if rs is not None:
        q["recurrent_states"] = rs
    logits = model.forward(input_ids=nxt, params=q)
    past += 1

text = tok.decode(torch.tensor(out))
print(f"PROMPT: {QUESTION!r}")
print(f"PROMPT_IDS: {ids.numel()} tokens")
print(f"ANSWER: {text!r}")
print("VERDICT:", "contains 'Paris'" if "paris" in text.lower() else "NO 'Paris' in output")
