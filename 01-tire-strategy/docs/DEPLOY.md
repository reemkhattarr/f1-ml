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

The demo (`tire_strategy.serving.demo`) needs three things at runtime: the
`tire_strategy` package, the trained artifacts (`artifacts/*.joblib`,
`metadata.json`), and race data.

**Important caveat (honest):** the demo currently pulls race data **live from
FastF1**, which is subject to a ~500-calls/hour rate limit. That is fine locally
(cached after first use) but fragile on an ephemeral Space. The recommended
deployment is to **pre-build a small parquet of a few showcase races and ship it
in the Space**, then have the demo read from that parquet instead of live FastF1.
This change is not yet implemented — it is the concrete next step for a hosted demo.

**Outline of the deployment (once the above is done):**
1. Create a Gradio Space on Hugging Face.
2. Include: the `src/tire_strategy` package (or `pip install` it), `artifacts/`
   (the trained models — these are gitignored in the main repo, so they must be
   added to the Space explicitly), a Space `app.py` that calls `build_demo().launch()`,
   and a `requirements.txt`.
3. Push to the Space's git remote.

I can scaffold the Space (app entry + requirements + the parquet-backed demo
variant) when you have a Hugging Face account ready.

## FastAPI service → Render / Railway (free tier)

The service is stateless (race state is passed in each request, nothing stored).
1. Add a `Dockerfile` (not yet written) that installs the package + `libgomp`
   (LightGBM's runtime) and runs `uvicorn tire_strategy.serving.app:app`.
2. Include the trained `artifacts/` in the image, or mount them.
3. Deploy on Render/Railway pointing at the Dockerfile.

> Status: the serving code and demo are complete and tested; the hosting wrappers
> (Space files, Dockerfile, parquet-backed demo) are the remaining deployment work,
> deferred until the target accounts are set up.
