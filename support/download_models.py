import argparse
from pathlib import Path

from huggingface_hub import snapshot_download

from common import load_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--with-llm", action="store_true")
    args = parser.parse_args()
    config, runtime = load_config(args.config)
    root = Path(args.config).resolve().parent / config.get("models_dir", "models")
    tokenizer_files = ["*token*", "vocab.json", "merges.txt", "*config.json", "*.jinja"]
    snapshot_download(config["embedding_model"], revision=config["embedding_revision"],
                      local_dir=root / "embedding",
                      allow_patterns=tokenizer_files + ["model.safetensors"])
    snapshot_download(runtime["model"], revision=runtime["revision"],
                      local_dir=root / "qwen_tokenizer", allow_patterns=tokenizer_files)
    if args.with_llm:
        snapshot_download(runtime["model"], revision=runtime["revision"],
                          local_dir=root / "qwen",
                          allow_patterns=tokenizer_files + ["*.safetensors", "*.safetensors.index.json"])


if __name__ == "__main__":
    main()
