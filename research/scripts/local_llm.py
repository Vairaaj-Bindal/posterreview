"""Local open-source LLM backend — zero monetary cost, engine-agnostic.

Two interchangeable engines behind one interface, so the SAME reviewer runs on:
  - "mlx"    — Apple Silicon (this Mac) via MLX. Small model, dev/testing.
  - "ollama" — anywhere Ollama runs (esp. a DGX Spark / any CUDA box) via its
               local HTTP API. Big model (e.g. llama3.3 70B), near-paid quality.

Because the design-metrics engine already extracts the poster's visual facts, the
review is a TEXT task — no vision model needed.

Engine selection (first match wins):
  1. POSTERREVIEW_ENGINE env ("mlx" | "ollama")
  2. an Ollama server reachable at OLLAMA_HOST (default 127.0.0.1:11434) → "ollama"
  3. else → "mlx"

Models (override via env):
  POSTERREVIEW_MLX_MODEL     default mlx-community/Qwen2.5-7B-Instruct-4bit
  POSTERREVIEW_OLLAMA_MODEL  default llama3.3          (pull it on the Spark)
  OLLAMA_HOST                default 127.0.0.1:11434
"""
from __future__ import annotations

import json
import os
import re
import urllib.request

MLX_MODEL = os.environ.get("POSTERREVIEW_MLX_MODEL", "mlx-community/Qwen2.5-7B-Instruct-4bit")
OLLAMA_MODEL = os.environ.get("POSTERREVIEW_OLLAMA_MODEL", "llama3.3")
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "127.0.0.1:11434").replace("http://", "").rstrip("/")

_MLX_CACHE = {}


def _ollama_up() -> bool:
    try:
        urllib.request.urlopen(f"http://{OLLAMA_HOST}/api/tags", timeout=1.5)
        return True
    except Exception:
        return False


def select_engine() -> str:
    env = os.environ.get("POSTERREVIEW_ENGINE")
    if env in ("mlx", "ollama"):
        return env
    return "ollama" if _ollama_up() else "mlx"


# ---------- engines ----------
def _mlx_generate(prompt, system, max_tokens) -> str:
    from mlx_lm import load, generate
    from mlx_lm.sample_utils import make_sampler
    if MLX_MODEL not in _MLX_CACHE:
        _MLX_CACHE[MLX_MODEL] = load(MLX_MODEL)
    model, tok = _MLX_CACHE[MLX_MODEL]
    msgs = ([{"role": "system", "content": system}] if system else []) + \
           [{"role": "user", "content": prompt}]
    chat = tok.apply_chat_template(msgs, add_generation_prompt=True)
    return generate(model, tok, prompt=chat, max_tokens=max_tokens,
                    sampler=make_sampler(temp=0.3), verbose=False)


def _ollama_generate(prompt, system, max_tokens) -> str:
    msgs = ([{"role": "system", "content": system}] if system else []) + \
           [{"role": "user", "content": prompt}]
    body = json.dumps({
        "model": OLLAMA_MODEL, "messages": msgs, "stream": False,
        "keep_alive": "30m",  # keep the big model resident so reviews don't cold-reload
        "options": {"temperature": 0.3, "num_predict": max_tokens},
    }).encode()
    req = urllib.request.Request(f"http://{OLLAMA_HOST}/api/chat", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.load(r)["message"]["content"]


def generate_text(prompt: str, system: str = "", max_tokens: int = 3000,
                  engine: str | None = None) -> str:
    engine = engine or select_engine()
    return (_ollama_generate if engine == "ollama" else _mlx_generate)(prompt, system, max_tokens)


# ---------- JSON helpers (small local models sometimes wrap/trail their JSON) ----------
def _extract_json(text: str):
    text = re.sub(r"```(json)?", "", text or "").strip()
    start = text.find("{")
    if start < 0:
        return None
    depth, instr, esc = 0, False, False
    for i in range(start, len(text)):
        c = text[i]
        if instr:
            esc = (c == "\\" and not esc)
            if c == '"' and not esc:
                instr = False
        elif c == '"':
            instr = True
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return None


def generate_json(prompt: str, system: str = "", max_tokens: int = 3500,
                  engine: str | None = None) -> dict | None:
    engine = engine or select_engine()
    raw = generate_text(prompt, system, max_tokens, engine)
    blob = _extract_json(raw)
    if blob:
        try:
            return json.loads(blob)
        except json.JSONDecodeError:
            pass
    # one repair retry
    fix = ("Return ONLY a single valid JSON object — no prose, no code fences. "
           "Fix and reprint this JSON:\n\n" + (raw or "")[:4000])
    blob = _extract_json(generate_text(fix, system, max_tokens, engine))
    if blob:
        try:
            return json.loads(blob)
        except json.JSONDecodeError:
            return None
    return None


def engine_info() -> str:
    e = select_engine()
    return f"{e} ({OLLAMA_MODEL} @ {OLLAMA_HOST})" if e == "ollama" else f"mlx ({MLX_MODEL})"


if __name__ == "__main__":
    print("engine:", engine_info())
    print(generate_text("Name one benefit of research posters in one sentence.", max_tokens=60))
