# Deploying the PosterReview web service (zero compute cost)

The app runs **on the DGX Spark** (where the 72B model lives) inside a hardened
Docker container, exposed to the internet through a Cloudflare tunnel. The Mac
isn't involved. Everything is free except (optionally) a custom domain.

## Architecture

```
public  →  Cloudflare edge  →  cloudflared (Spark host)  →  127.0.0.1:5000
                                                              │  (published, localhost-only)
                                                     ┌────────▼─────────┐
                                                     │ Docker container │  non-root, read-only rootfs,
                                                     │  PosterReview app│  cap-drop ALL, no host mounts,
                                                     │  (parse + review)│  mem/cpu/pids caps
                                                     └────────┬─────────┘
                                                     host.docker.internal:11434
                                                              │
                                                       Ollama (qwen2.5:72b) on the Spark GPU
```

Untrusted PDF parsing is isolated **twice**: in a resource-limited subprocess
(`review_worker.py`) *inside* a locked-down container. Ollama and the app port
are bound to the Spark's localhost only; the tunnel is the sole ingress.

## Run / update it (on the Spark)

```bash
cd ~/posterreview && git pull && bash app/deploy_spark.sh   # build + (re)start container
# public tunnel (URL lands in ~/cf.log):
nohup ~/bin/cloudflared tunnel --url http://127.0.0.1:5000 >~/cf.log 2>&1 &
grep -oE 'https://[a-z0-9-]+\.trycloudflare\.com' ~/cf.log | head -1
```

To stop the tunnel: `pkill -x cloudflared` (use `-x`, not `-f` — `-f cloudflared`
matches your own SSH command line and kills your shell). Container: `docker
rm -f posterreview`.

## Security posture

- Container: non-root (`appuser` uid 10001), `--read-only` rootfs (writes only to
  small tmpfs for uploads/tmp/cache), `--cap-drop ALL`, `--security-opt
  no-new-privileges`, `--memory`/`--cpus`/`--pids-limit` caps, no host FS or GPU.
- Input: 30 MB cap, `%PDF` magic-byte check, per-review wall-clock timeout with
  process-group kill, Pillow decompression-bomb cap, subprocess RLIMITs.
- Abuse: per-IP rate limit (real IP via `CF-Connecting-IP`), global concurrency cap.
- Web: CSP + `X-Frame-Options: DENY` + nosniff + `Referrer-Policy`, generic client
  errors (details server-side only), HTML-escaped output + URL-scheme allowlist.
- Ingress: app binds localhost; cloudflared is outbound-only (home IP stays private).

## Stable URL / custom domain (the one paid piece)

The `trycloudflare.com` URL is a **quick tunnel** — free but ephemeral (changes on
cloudflared restart, dies on reboot). For a durable `posterreview.ai`:

1. Register `posterreview.ai` (~$70–100/yr for `.ai` — the only real cost;
   compute stays free).
2. Add it to a free Cloudflare account.
3. Replace the quick tunnel with a **named tunnel** bound to the hostname
   (`cloudflared tunnel create` + a `config.yml` + DNS route), run as a
   systemd/user service so it auto-starts and survives reboots.
