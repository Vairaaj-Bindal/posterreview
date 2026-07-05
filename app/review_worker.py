"""Isolated review worker — runs ONE review in its own process under strict limits.

The app spawns this per upload so untrusted PDF parsing (PyMuPDF / RapidOCR / PIL)
never runs in the web server's process. A parser crash, hang, or memory bomb is
contained here: hard memory cap, CPU cap, and the parent enforces a wall-clock
timeout and kills the process group. Output is a single JSON object on stdout.

argv: <pdf_path> <use_arxiv 0|1>
"""
import json
import os
import resource
import sys

# --- hard resource limits (defense against decompression bombs / runaway OCR) ---
_MEM_BYTES = int(os.environ.get("PR_WORKER_MEM_MB", "4096")) * 1024 * 1024
_CPU_SECS = int(os.environ.get("PR_WORKER_CPU_SECS", "240"))
try:
    resource.setrlimit(resource.RLIMIT_AS, (_MEM_BYTES, _MEM_BYTES))
    resource.setrlimit(resource.RLIMIT_CPU, (_CPU_SECS, _CPU_SECS))
    resource.setrlimit(resource.RLIMIT_FSIZE, (64 * 1024 * 1024, 64 * 1024 * 1024))
    resource.setrlimit(resource.RLIMIT_NOFILE, (256, 256))
except (ValueError, OSError):
    pass

# cap Pillow decompression-bomb surface before anything imports/uses PIL
try:
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = 64_000_000  # ~64MP; raises on larger
except Exception:
    pass

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "research", "scripts"))


def main():
    pdf_path, use_arxiv = sys.argv[1], sys.argv[2] == "1"
    import review as review_mod
    out = review_mod.build_review(pdf_path, backend="local", use_arxiv=use_arxiv)
    # strip anything not needed by the UI (keeps payload small, avoids leaking internals)
    slim = {
        "design_score": out.get("design_score"),
        "provisional_score": out.get("provisional_score"),
        "review": out.get("review"),
        "related_work": [
            {"title": p.get("title", ""), "url": p.get("url", ""), "year": p.get("year", "")}
            for p in (out.get("related_work") or [])
        ],
    }
    sys.stdout.write(json.dumps(slim))


if __name__ == "__main__":
    main()
