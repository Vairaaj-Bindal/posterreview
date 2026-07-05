"""PosterReview web app — upload a poster, get a grounded review + design score.

Zero-cost: runs the local pipeline (metrics + design score + arXiv + local LLM
review via MLX or the Spark's Ollama). Reviews take a couple minutes, so the
request is async: upload starts a background job, the page polls for the result.
"""
import os
import pathlib
import threading
import traceback
import uuid

from flask import Flask, jsonify, render_template, request

# local pipeline lives in research/scripts
ROOT = pathlib.Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT / "research" / "scripts"))

# default to the Spark's Ollama over the SSH tunnel; falls back to MLX if absent
os.environ.setdefault("OLLAMA_HOST", "127.0.0.1:11434")
os.environ.setdefault("POSTERREVIEW_OLLAMA_MODEL", "qwen2.5:72b")

import review as review_mod
import local_llm

app = Flask(__name__)
UPLOADS = ROOT / "app" / "uploads"
UPLOADS.mkdir(parents=True, exist_ok=True)

JOBS = {}  # job_id -> {"status": "running"|"done"|"error", "result"/"error"}


def _run(job_id, pdf_path, use_arxiv):
    try:
        out = review_mod.build_review(str(pdf_path), backend="local", use_arxiv=use_arxiv)
        JOBS[job_id] = {"status": "done", "result": out}
    except Exception as e:
        traceback.print_exc()
        JOBS[job_id] = {"status": "error", "error": str(e)}
    finally:
        try:
            pdf_path.unlink()
        except OSError:
            pass


@app.route("/")
def index():
    return render_template("index.html", engine=local_llm.engine_info())


@app.route("/review", methods=["POST"])
def start_review():
    f = request.files.get("poster")
    if not f or not f.filename.lower().endswith(".pdf"):
        return jsonify({"error": "Please upload a poster PDF."}), 400
    job_id = uuid.uuid4().hex[:12]
    pdf_path = UPLOADS / f"{job_id}.pdf"
    f.save(pdf_path)
    use_arxiv = request.form.get("arxiv", "on") == "on"
    JOBS[job_id] = {"status": "running"}
    threading.Thread(target=_run, args=(job_id, pdf_path, use_arxiv), daemon=True).start()
    return jsonify({"job_id": job_id})


@app.route("/status/<job_id>")
def status(job_id):
    return jsonify(JOBS.get(job_id, {"status": "unknown"}))


if __name__ == "__main__":
    print(f"PosterReview — engine: {local_llm.engine_info()}")
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)
