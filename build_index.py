import argparse
import json
import platform
import resource
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from common import load_config, read_jsonl, write_jsonl
from retrieval import encode, make_chunks, normalize_math


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    config, runtime = load_config(args.config)
    models = Path(args.config).resolve().parent / config.get("models_dir", "models")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(config["cpu_threads"])
    corpus = read_jsonl(args.corpus)
    loading = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(models / "embedding", local_files_only=True)
    model = AutoModel.from_pretrained(models / "embedding", local_files_only=True).cpu().eval()
    load_seconds = time.perf_counter() - loading
    parameter_count = sum(p.numel() for p in model.parameters())
    chunks = make_chunks(corpus, tokenizer, config)
    passages = ["passage: " + c["section"] + "\n" + normalize_math(c["text"]) for c in chunks]
    embedding_started = time.perf_counter()
    vectors = encode(passages, tokenizer, model, config["embedding_batch_size"])
    embedding_seconds = time.perf_counter() - embedding_started
    write_jsonl(output / "corpus.jsonl", corpus)
    write_jsonl(output / "chunks.jsonl", chunks)
    np.save(output / "vectors.npy", vectors)
    model.save_pretrained(output / "embedding")
    tokenizer.save_pretrained(output / "embedding")
    qwen = AutoTokenizer.from_pretrained(models / "qwen_tokenizer", local_files_only=True)
    qwen.save_pretrained(output / "qwen_tokenizer")
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak = rss / (1024 ** 3 if platform.system() == "Darwin" else 1024 ** 2)
    manifest = {"config": config, "runtime": runtime,
                "blocks": len(corpus), "chunks": len(chunks), "embedding_parameters": parameter_count,
                "build_seconds": time.perf_counter() - started, "peak_ram_gib": peak,
                "load_seconds": load_seconds, "embedding_seconds": embedding_seconds,
                "index_bytes": sum(p.stat().st_size for p in output.rglob("*") if p.is_file()),
                "hardware": {"platform": platform.platform(), "processor": platform.processor(),
                             "cpu_count": __import__("os").cpu_count()},
                "python": platform.python_version()}
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: manifest[k] for k in ("blocks", "chunks", "build_seconds", "peak_ram_gib", "index_bytes")}))


if __name__ == "__main__":
    main()
