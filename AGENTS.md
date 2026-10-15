# AGENTS.md

Static GitHub Pages site: `index.html` (three.js from a CDN) renders `data/tensors/<publisher>__<model>.json`, listed in `data/tensors/index.json`.

`fetch_tensors.py` builds that data from the safetensors headers of the models in the hf-top-models ranking (`MODELS_URL`). Constants at the top. Rerunning only fetches missing repos; delete a model's json to refetch it.

**RESPECT THE HUGGING FACE API RATE LIMITS** (https://huggingface.co/docs/hub/rate-limits). Never run two fetchers at once, never in a loop, keep `WORKERS` low.

Record format: `{id, rank, likes, n_files, n_tensors, tensors: {name: [dtype, shape, count]}, error?}`. Numeric segments after the layer index (MoE experts) collapse to `*` with a count. Tensors in subfolders (diffusers) get the folder as a prefix.

Dev telemetry: on localhost the page POSTs console errors and long tasks to `/log`. Any static server works without it.
