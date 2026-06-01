# Deployment

Project 1 has two deployable surfaces: the **FastAPI service** and the **Gradio
demo**. Both run locally today; hosting them is a manual step that needs your own
accounts (Hugging Face / Render), exactly like the GitHub remote did.

## Run locally

```bash
brew install libomp        # macOS: LightGBM runtime (one-time)
just install
just train                 # produces artifacts/ (gitignored): models + metadata
just serve                 # FastAPI at http://localhost:8000  (docs at /docs)
just demo                  # Gradio demo in the browser
```

## Gradio demo → Hugging Face Spaces (free)

The demo (`tire_strategy.serving.demo`) reads the **pre-built dataset parquet**, not
live FastF1, so it has no rate-limit fragility on an ephemeral Space. Everything
needed is assembled by a build script.

**1. Build the deployable Space folder** (after `just data` + `just train`):

```bash
just build-space        # -> 01-tire-strategy/deploy/space_build/  (~6 MB, gitignored)
```

This produces a self-contained folder mirroring the repo layout (so the package's
path config resolves), bundling the package source, the trained artifacts
(`*.joblib` + `metadata.json`), the tuned config, the dataset parquet, and the
Space's `app.py` / `requirements.txt` / `README.md` (with the Gradio SDK header).
It is verified to load the models and predict from its own layout.

**2. Create a Gradio Space** on Hugging Face (one-time):

```bash
pip install huggingface_hub           # if needed
huggingface-cli login                 # paste a token from hf.co/settings/tokens
huggingface-cli repo create f1-tire-strategy --type space --space_sdk gradio
```

**3. Push the assembled folder to the Space:**

```bash
cd 01-tire-strategy/deploy/space_build
git init && git remote add origin https://huggingface.co/spaces/<user>/f1-tire-strategy
git add -A && git commit -m "Deploy F1 tire-strategy demo"
git push -u origin main
```

The Space builds from `requirements.txt` and launches `app.py`. The demo source
files live in `deploy/hf_space/`; the build script copies them.

> Steps 2–3 need your Hugging Face account — happy to walk through them together.

## FastAPI service → Render / Railway (free tier)

The service is stateless (race state is passed in each request, nothing stored).
1. Add a `Dockerfile` (not yet written) that installs the package + `libgomp`
   (LightGBM's runtime) and runs `uvicorn tire_strategy.serving.app:app`.
2. Include the trained `artifacts/` in the image, or mount them.
3. Deploy on Render/Railway pointing at the Dockerfile.

> Status: the serving code and demo are complete and tested; the hosting wrappers
> (Space files, Dockerfile, parquet-backed demo) are the remaining deployment work,
> deferred until the target accounts are set up.
