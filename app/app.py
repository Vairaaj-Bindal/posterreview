"""PosterReview web app — upload a poster, get a grounded review + design score.

Hardened for public exposure (behind a Cloudflare tunnel):
  - untrusted PDF parsing runs in an isolated subprocess (review_worker.py) with
    memory/CPU/file limits and a wall-clock timeout enforced by the parent;
  - uploads are size-capped and magic-byte-validated as PDFs;
  - per-IP rate limiting + a global concurrency cap;
  - generic client errors (details logged server-side only);
  - security headers; binds to localhost only (the tunnel is the sole ingress).
"""
import json
import logging
import os
import pathlib
import signal
import subprocess
import sys
import threading
import time
import uuid

from flask import Flask, jsonify, render_template, request

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research" / "scripts"))
os.environ.setdefault("OLLAMA_HOST", "127.0.0.1:11434")
os.environ.setdefault("POSTERREVIEW_OLLAMA_MODEL", "qwen2.5:72b")
import local_llm  # noqa: E402  (only for engine_info on the landing page)

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("posterreview")

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 30 * 1024 * 1024  # 30 MB hard upload cap

UPLOADS = ROOT / "app" / "uploads"
UPLOADS.mkdir(parents=True, exist_ok=True)
WORKER = str(ROOT / "app" / "review_worker.py")

MAX_ACTIVE = int(os.environ.get("POSTERREVIEW_MAX_ACTIVE", "2"))     # concurrent reviews
REVIEW_TIMEOUT = int(os.environ.get("POSTERREVIEW_TIMEOUT", "360"))  # seconds per review
RATE_N = int(os.environ.get("POSTERREVIEW_RATE_N", "6"))             # reviews per window / IP
RATE_WINDOW = int(os.environ.get("POSTERREVIEW_RATE_WINDOW", "1800"))  # 30 min

JOBS = {}
_active = 0
_lock = threading.Lock()
_ip_hits = {}  # ip -> [timestamps]


def client_ip():
    # behind the Cloudflare tunnel the real client is in CF-Connecting-IP;
    # remote_addr is just the local cloudflared. Fall back safely.
    return (request.headers.get("CF-Connecting-IP")
            or (request.headers.get("X-Forwarded-For", "").split(",")[0].strip())
            or request.remote_addr or "?")


def rate_ok(ip) -> bool:
    now = time.monotonic()
    with _lock:
        hits = [t for t in _ip_hits.get(ip, []) if now - t < RATE_WINDOW]
        if len(hits) >= RATE_N:
            _ip_hits[ip] = hits
            return False
        hits.append(now)
        _ip_hits[ip] = hits
        if len(_ip_hits) > 5000:  # bound memory
            _ip_hits.clear()
        return True


def _run(job_id, pdf_path, use_arxiv):
    global _active
    proc = None
    try:
        proc = subprocess.Popen(
            [sys.executable, WORKER, str(pdf_path), "1" if use_arxiv else "0"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            start_new_session=True,  # own process group, so we can kill children too
        )
        out, err = proc.communicate(timeout=REVIEW_TIMEOUT)
        if proc.returncode != 0:
            log.warning("worker failed rc=%s job=%s err=%s", proc.returncode, job_id, err[-400:])
            JOBS[job_id] = {"status": "error"}
        else:
            JOBS[job_id] = {"status": "done", "result": json.loads(out)}
    except subprocess.TimeoutExpired:
        log.warning("worker timeout job=%s", job_id)
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except (ProcessLookupError, AttributeError, TypeError):
            pass
        JOBS[job_id] = {"status": "error"}
    except Exception:
        log.exception("review dispatch failed job=%s", job_id)
        JOBS[job_id] = {"status": "error"}
    finally:
        with _lock:
            _active -= 1
        try:
            pdf_path.unlink()
        except OSError:
            pass
        if len(JOBS) > 300:
            for k in [k for k, v in list(JOBS.items())[:150] if v.get("status") != "running"]:
                JOBS.pop(k, None)


@app.after_request
def secure_headers(resp):
    resp.headers["X-Content-Type-Options"] = "nosniff"
    resp.headers["X-Frame-Options"] = "DENY"
    resp.headers["Referrer-Policy"] = "no-referrer"
    resp.headers["Content-Security-Policy"] = (
        "default-src 'self'; style-src 'self' 'unsafe-inline'; "
        "script-src 'self' 'unsafe-inline'; img-src 'self' data:; "
        "connect-src 'self'; base-uri 'none'; form-action 'self'; frame-ancestors 'none'")
    return resp


@app.route("/")
def index():
    return render_template("index.html", engine=local_llm.engine_info())


@app.route("/review", methods=["POST"])
def start_review():
    global _active
    ip = client_ip()
    f = request.files.get("poster")
    if not f or not (f.filename or "").lower().endswith(".pdf"):
        return jsonify({"error": "Please upload a poster PDF."}), 400
    head = f.stream.read(5); f.stream.seek(0)
    if head[:4] != b"%PDF":
        return jsonify({"error": "That doesn't look like a valid PDF."}), 400
    if not rate_ok(ip):
        return jsonify({"error": "Rate limit reached — please try again later."}), 429
    with _lock:
        if _active >= MAX_ACTIVE:
            return jsonify({"error": "The reviewer is busy right now — try again in a couple minutes."}), 503
        _active += 1
    job_id = uuid.uuid4().hex
    pdf_path = UPLOADS / f"{job_id}.pdf"
    try:
        f.save(pdf_path)
        JOBS[job_id] = {"status": "running"}
        threading.Thread(target=_run, args=(job_id, pdf_path, request.form.get("arxiv", "on") == "on"),
                         daemon=True).start()
    except Exception:  # never leak the concurrency slot if setup fails before the thread runs
        with _lock:
            _active -= 1
        log.exception("failed to start review job=%s", job_id)
        return jsonify({"error": "Could not process the upload."}), 500
    log.info("review started job=%s ip=%s", job_id, ip)
    return jsonify({"job_id": job_id})


@app.route("/status/<job_id>")
def status(job_id):
    if not job_id.isalnum() or len(job_id) > 40:
        return jsonify({"status": "unknown"}), 400
    j = JOBS.get(job_id, {"status": "unknown"})
    # never leak internal error text to the client
    return jsonify({"status": j["status"], "result": j.get("result")}
                   if j["status"] == "done" else {"status": j["status"]})


@app.errorhandler(413)
def too_large(_):
    return jsonify({"error": "File too large (max 30 MB)."}), 413


if __name__ == "__main__":
    print(f"PosterReview — engine: {local_llm.engine_info()}")
    # production server if available; dev server otherwise. Bind localhost only —
    # the Cloudflare tunnel is the sole public ingress.
    # 127.0.0.1 by default (tunnel is sole ingress); the container sets 0.0.0.0
    # since its published port is itself bound to the host's localhost only.
    bind = os.environ.get("POSTERREVIEW_BIND", "127.0.0.1")
    try:
        from waitress import serve
        serve(app, host=bind, port=5000, threads=8)
    except ImportError:
        app.run(host=bind, port=5000, debug=False, threaded=True)
