"""Tensor structure (name, dtype, shape) of the TOP most-liked models -> data/tensors/<publisher>__<model>.json

Stored compact: numeric segments after the layer index (MoE experts) collapse to "*" with a count,
tensors in subfolders (diffusers) get the folder as prefix. data/tensors/index.json lists what's there.

WARNING: RESPECT THE HUGGING FACE API RATE LIMITS. https://huggingface.co/docs/hub/rate-limits
Repos go one at a time: one file listing, then the .safetensors headers (one range request each) with WORKERS in parallel.
On HTTP 429 the request waits RETRY_WAIT and retries instead of recording an error.
Already fetched repos are skipped, so rerunning only fetches what's missing.

Reads the model list (the ranking) from MODELS_URL, published by https://github.com/franciscocarloserra/hf-top-models
"""
import json, pathlib, time, urllib.request
from concurrent.futures import ThreadPoolExecutor
from huggingface_hub import list_repo_files, parse_safetensors_file_metadata

TOP = 300
MODELS_URL = "https://franciscocarloserra.github.io/hf-top-models/data/models.json"
SLEEP = 0.1  # seconds each worker waits between requests
WORKERS = 3  # parallel header requests within a repo (logged-in rate limits)
RETRY_WAIT = 60  # seconds to wait after a 429

DATA = pathlib.Path(__file__).with_name("data")
OUT = DATA / "tensors"
OUT.mkdir(exist_ok=True)


def compact(files):
    """{file: {name: [dtype, shape]}} -> {name: [dtype, shape, count]}"""
    out = {}
    for f, tensors in files.items():
        folder = f.rsplit("/", 1)[0] + "/" if "/" in f else ""
        for name, (dtype, shape) in tensors.items():
            parts, seen = name.split("."), False
            for i, p in enumerate(parts):
                if p.isdigit():
                    parts[i], seen = (p if not seen else "*"), True
            key = folder + ".".join(parts)
            if key not in out:
                out[key] = [dtype, shape, 0]
            if out[key][1] == shape:
                out[key][2] += 1
    return out


def write_index():
    index = []
    for path in sorted(OUT.glob("*__*.json")):
        rec = json.loads(path.read_text())
        if "files" in rec:  # raw record from an older run: compact in place
            rec["n_files"], rec["n_tensors"] = len(rec["files"]), sum(len(t) for t in rec["files"].values())
            rec["tensors"] = compact(rec.pop("files"))
            path.write_text(json.dumps(rec, separators=(",", ":")))
        index.append({"file": path.stem, "id": rec["id"], "rank": rec["rank"], "error": rec.get("error")})
    (OUT / "index.json").write_text(json.dumps(sorted(index, key=lambda r: r["rank"]), indent=1))


def header(repo, f):
    while True:
        time.sleep(SLEEP)
        try:
            return parse_safetensors_file_metadata(repo, f)
        except Exception as e:
            if getattr(getattr(e, "response", None), "status_code", None) != 429:
                raise
            print("429, waiting", RETRY_WAIT, "s", flush=True)
            time.sleep(RETRY_WAIT)


write_index()
models = sorted(json.load(urllib.request.urlopen(MODELS_URL))["models"], key=lambda m: -m["likes"])[:TOP]
for rank, m in enumerate(models, 1):
    path = OUT / (m["id"].replace("/", "__") + ".json")
    if path.exists():
        continue
    rec = {"id": m["id"], "rank": rank, "likes": m["likes"], "files": {}}  # files is replaced by tensors below
    try:
        files = [f for f in list_repo_files(m["id"]) if f.endswith(".safetensors")]
        with ThreadPoolExecutor(WORKERS) as pool:
            for f, meta in zip(files, pool.map(lambda f: header(m["id"], f), files)):
                rec["files"][f] = {k: [v.dtype, v.shape] for k, v in meta.tensors.items()}
        if not files:
            rec["error"] = "no .safetensors files"
    except Exception as e:
        rec["error"] = f"{type(e).__name__}: {e}"[:300]
    n = sum(len(t) for t in rec["files"].values())
    rec["n_files"], rec["n_tensors"], rec["tensors"] = len(rec["files"]), n, compact(rec.pop("files"))
    path.write_text(json.dumps(rec, separators=(",", ":")))
    print(rank, m["id"], rec["n_files"], "files", n, "tensors", rec.get("error", ""), flush=True)
    write_index()
    time.sleep(SLEEP)

