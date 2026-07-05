# Hardened container for the PosterReview web app.
# Isolates untrusted-PDF parsing off the host: non-root, no host mounts,
# minimal deps. The review LLM runs on the host's Ollama (reached via
# host.docker.internal); this container never has GPU or host FS access.
FROM python:3.12-slim

# minimal system libs PyMuPDF / onnxruntime need at runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 libgl1 && rm -rf /var/lib/apt/lists/*

WORKDIR /srv
COPY app/requirements.txt /srv/app/requirements.txt
RUN pip install --no-cache-dir -r /srv/app/requirements.txt

# only the code the server needs — not the research data/notebooks
COPY app/ /srv/app/
COPY research/scripts/ /srv/research/scripts/

# non-root, writable cache + uploads dirs (image itself runs read-only)
RUN useradd -m -u 10001 appuser \
 && mkdir -p /srv/app/uploads /home/appuser/.cache \
 && chown -R appuser:appuser /srv/app/uploads /home/appuser
USER appuser

ENV POSTERREVIEW_BIND=0.0.0.0 \
    OLLAMA_HOST=host.docker.internal:11434 \
    POSTERREVIEW_OLLAMA_MODEL=qwen2.5:72b \
    HOME=/home/appuser

EXPOSE 5000
CMD ["python", "/srv/app/app.py"]
