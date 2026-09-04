# Deploying

The browser app is a **static site that does all inference client-side**.
There is no server, no API, and no per-image cost. That makes deployment
almost trivial — except for one thing: the model is 210 MB, and it must not
live on the same host as the site.

## The shape of it

```
  ┌──────────────────────────┐        ┌────────────────────────────┐
  │  Vercel                  │        │  Hugging Face Hub          │
  │  ~25 MB static           │        │  210 MB model              │
  │  HTML + JS + ORT wasm    │◀──────▶│  free, CDN, no bandwidth   │
  │  free, custom domain     │  fetch │  cap on public repos       │
  └──────────────────────────┘        └────────────────────────────┘
                    ▲
                    │  one 210 MB download, then Cache API
                    ▼
              the visitor's browser — where every image is
              actually processed. Nothing is ever uploaded.
```

### Why the model is not on Vercel

Two independent reasons, either one fatal:

- **Vercel Hobby caps uploads at 100 MB per file.** A 210 MB model cannot be
  deployed there at all.
- **Hobby gives 100 GB of bandwidth a month.** Even if the model fit, 210 MB
  per cold visitor means the site dies at roughly **475 first-time visitors**.
  Exceed it and deployments pause.

Hugging Face has neither limit for public repos, is CDN-backed, sends
permissive CORS headers, and is where the upstream weights already live.
A GitHub Release asset (2 GB limit, CORS-enabled) works equally well.

## Steps

### 1. Build the model

Needs the torch extra (~2.5 GB) — do this once, on a machine you don't mind
filling up.

```bash
uv pip install -e ".[torch]"
python scripts/export_onnx.py --model birefnet-lite --web
```

The `--web` flag is **not** optional: it rewrites deformable convolutions as
`grid_sample`, because onnxruntime-web has no `DeformConv` kernel. Without it
the browser fails at session creation.

Then merge the external-data sidecar into one file:

```bash
python -c "import onnx; m=onnx.load('weights/birefnet-lite-web.onnx'); \
onnx.save_model(m, 'birefnet-lite-web-merged.onnx', save_as_external_data=False)"
```

### 2. Host the model

```bash
.venv/bin/hf auth login
.venv/bin/hf upload <you>/bgremover-onnx \
  weights/birefnet-lite-web-merged.onnx birefnet-lite-web.onnx \
  --repo-type=model
```

Mark the repo public. Note in its card that the weights derive from
[BiRefNet_lite](https://huggingface.co/ZhengPeng7/BiRefNet_lite) (MIT) — see
[NOTICE](NOTICE).

### 3. Deploy the app

**`.vercelignore` is mandatory.** The Vercel CLI does not read `.gitignore`,
so without it the CLI tries to upload the 210 MB model plus 78 MB of ORT
runtime — 402 MB in total — and dies on the per-file cap:

```
DeploymentError: File size limit exceeded (100 MB)
```

With [`web/.vercelignore`](web/.vercelignore) the upload is 0.3 MB, and the
ORT runtime is regenerated on Vercel by the `prebuild` hook.

Point Vercel at the repo with **root directory `web`**. It picks up
[`web/vercel.json`](web/vercel.json) automatically. Set one environment
variable:

```
VITE_MODEL_URL = https://huggingface.co/<you>/bgremover-onnx/resolve/main/birefnet-lite-web.onnx
```

Or from the CLI:

```bash
cd web && npx vercel --prod
```

### Deployment Protection is on by default

The generated `*-yhechens-projects.vercel.app` URL 302s to Vercel SSO. The
short alias — `https://<project>-<hash>.vercel.app` — is public. If you want
the team URL public too, turn off Settings → Deployment Protection → Vercel
Authentication.

### `VITE_MODEL_URL` is a build-time variable

Vite inlines `import.meta.env` at build time, so setting the env var in the
Vercel dashboard does **not** affect the already-built site. Set it, then
redeploy.

## The headers matter

[`web/vercel.json`](web/vercel.json) sets:

```
Cross-Origin-Opener-Policy:   same-origin
Cross-Origin-Embedder-Policy: credentialless
```

`credentialless` rather than `require-corp` is a deliberate choice, and the
only value that satisfies both constraints at once:

- Threaded WASM needs `SharedArrayBuffer`, which needs cross-origin
  isolation, which needs COOP **and** COEP. Without them the WASM fallback
  drops to a single thread and becomes unusably slow.
- But `require-corp` **blocks** cross-origin fetches that lack a CORP header
  — which would block the model on Hugging Face.

`credentialless` keeps isolation while allowing the no-cors fetch. Verified:
`crossOriginIsolated === true` with the model loading off-origin.

The dev server sets the same headers, so a bug cannot hide until deploy.

## What about the T14?

**You don't need it for this.** The browser build has no server component, so
a laptop adds nothing and would only add a failure mode — a public service off
a laptop needs an always-on box plus a tunnel, and the hosted-inference path
is precisely the part that costs real money at scale. That is why remove.bg
ended up inside Canva.

Where the T14 genuinely helps:

- **Running the ONNX export.** It needs the 2.5 GB torch extra and a chunk of
  RAM. Do it there, upload the artefact, keep it off your other machines.
- **Batch work.** `bgremover batch ./photos ./cutouts` on the CLI, at full
  resolution with `--edge refine`, with no tab-memory ceiling.
- **Build-order step 4**, the self-hosted high-resolution API, if you ever
  want it for yourself. Behind Tailscale, not on the public internet.

## Known deployment gotchas

- **Do not quantise to int8.** It is smaller (159 MB) and faster and matches
  fp32 to IoU 0.9948 *on native CPU* — but on ORT-web's WebGPU path it
  returns a near-all-foreground mask. Measured `fg = 0.974` against an
  expected `0.443`. fp32 is what ships.
- **fp16 conversion produces a malformed graph** (`Type Error: tensor(float16)
  of output arg`) when `GridSample` is block-listed. Worth another attempt —
  it would halve the download — but it is not a one-liner.
- **`web/public/models/` and `web/public/ort/` are gitignored.** `npm run
  copy:ort` runs automatically via `predev`/`prebuild`; the model does not,
  which is exactly why `VITE_MODEL_URL` exists.
- **Cold start is the whole first impression.** 210 MB, reported as real byte
  progress and then stored via the Cache API so a second visit is instant.


## Current deployment

| | |
|---|---|
| Live | <https://backgroundremover-theta.vercel.app> |
| Project | `yhechens-projects/backgroundremover` |
| Status | App deployed and serving. **Model not yet hosted**, so the page loads and then reports `model fetch failed: 404`. |

Remaining: upload the model (step 2), set `VITE_MODEL_URL`, redeploy.
