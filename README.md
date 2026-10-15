# 3D Model Weights

**Live: https://franciscocarloserra.github.io/3d-model-weights/**

The weight matrices of the most-liked Hugging Face models, drawn in 3D. Every 2D tensor is a slab with sides ∝ √(out) and √(in). Layers go from input on the left to output on the right. The model list is the top 300 by likes from [hf-top-models](https://github.com/franciscocarloserra/hf-top-models) ([site](https://franciscocarloserra.github.io/hf-top-models/)).

Click a slab (or a row in the tensor panel on the right) to select that subcomponent in every layer. Shift-click adds to the selection, Esc clears it. Hovering a row or a stack header in the panel highlights it.

## Run

    python3 fetch_tensors.py         # fill data/tensors/, skips models already there
    python3 -m http.server           # open http://localhost:8000

`fetch_tensors.py` reads only the safetensors headers, one range request per file, and never downloads the weights. **Respect the Hugging Face API rate limits** (https://huggingface.co/docs/hub/rate-limits). It handles one repo at a time with `WORKERS` header requests in parallel, and waits and retries on 429. It needs `huggingface_hub`, and runs best logged in (`hf auth login`).

## Caveats

- Gated repos and GGUF-only repos have no data and are hidden from the list.
- Shapes come from the headers. Sizes are params × dtype bytes, i.e. the file size, not inference memory.
- Per-expert MoE tensors are stored once with a count (`experts.*.w1 ×384`). Fused 3D expert tensors are drawn the same way.
- Layer grouping and the attention/mlp colors are guessed from tensor names. Unusual architectures can come out odd.

## License

Code: MIT, see `LICENSE`. The data in `data/tensors/` is tensor names, shapes and dtypes read from public Hugging Face repos. It belongs to the original model owners and stays under each model's own license. No weights are included. Not affiliated with Hugging Face or any model publisher.
