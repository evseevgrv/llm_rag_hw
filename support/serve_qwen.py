import argparse
import json
import resource
import threading
import time
import uuid
from pathlib import Path

import torch
from fastapi import FastAPI, HTTPException, Request
from transformers import (AutoTokenizer, BitsAndBytesConfig, LogitsProcessorList,
                          Qwen3_5ForConditionalGeneration, set_seed)
import uvicorn

from common import ROOT, load_config
from .generation import PresencePenalty


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--journal", default=str(ROOT / "workspace" / "artifacts" / "endpoint_log.jsonl"))
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required; CPU/MPS generation is not enabled")
    config, runtime = load_config(args.config)
    path = Path(args.config).resolve().parent / config.get("models_dir", "models") / "qwen"
    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    quantization = dict(runtime["quantization"])
    quantization["bnb_4bit_compute_dtype"] = torch.float16
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        path, local_files_only=True, quantization_config=BitsAndBytesConfig(**quantization),
        device_map={"": 0}, dtype=torch.float16, attn_implementation="eager").eval()
    loaded = time.perf_counter() - started
    warmup = time.perf_counter()
    inputs = tokenizer("Здравствуйте", return_tensors="pt").to("cuda")
    with torch.inference_mode():
        model.generate(**inputs, max_new_tokens=8, do_sample=False,
                       pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
    torch.cuda.synchronize()
    startup = {"load_seconds": loaded, "warmup_seconds": time.perf_counter() - warmup,
               "gpu": torch.cuda.get_device_name(0),
               "peak_vram_gib": torch.cuda.max_memory_allocated() / 1024 ** 3,
               "revision": runtime["revision"], "runtime": runtime}
    journal = Path(args.journal)
    journal.parent.mkdir(parents=True, exist_ok=True)
    journal.with_name("server_resources.json").write_text(json.dumps(startup, ensure_ascii=False, indent=2) + "\n")
    lock = threading.Lock()
    app = FastAPI()

    @app.get("/health")
    def health():
        return {"ready": True, "model": runtime["model"], "revision": runtime["revision"]}

    @app.post("/v1/chat/completions")
    def complete(request: Request, payload: dict):
        qid = request.headers["x-question-id"]
        submission = request.headers["x-submission-id"]
        with lock:
            started = time.perf_counter()
            set_seed(runtime["generation"]["seed"])
            prompt = tokenizer.apply_chat_template(payload["messages"], tokenize=False,
                                                    add_generation_prompt=True,
                                                    enable_thinking=runtime["enable_thinking"])
            batch = tokenizer(prompt, add_special_tokens=False, return_tensors="pt").to("cuda")
            input_tokens = batch["input_ids"].shape[-1]
            if input_tokens > runtime["max_input_tokens"] or input_tokens + payload["max_tokens"] > runtime["max_request_tokens"]:
                raise HTTPException(400, "Token budget exceeded")
            with torch.inference_mode():
                output = model.generate(**batch, max_new_tokens=payload["max_tokens"], do_sample=True,
                                        temperature=payload["temperature"], top_p=payload["top_p"],
                                        top_k=payload["top_k"], min_p=payload["min_p"],
                                        repetition_penalty=payload["repetition_penalty"],
                                        logits_processor=LogitsProcessorList([
                                            PresencePenalty(input_tokens, payload["presence_penalty"])]),
                                        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
            generated = output[0, input_tokens:]
            count = generated.numel()
            stop_ids = model.generation_config.eos_token_id
            stop_ids = [stop_ids] if isinstance(stop_ids, int) else stop_ids or [tokenizer.eos_token_id]
            reason = "stop" if generated[-1].item() in stop_ids else "length"
            response_id = "chatcmpl-" + uuid.uuid4().hex
            record = {"response_id": response_id, "id": qid, "submission_id": submission,
                      "input_tokens": input_tokens, "output_tokens": count,
                      "max_tokens": payload["max_tokens"], "finish_reason": reason,
                      "elapsed_ms": (time.perf_counter() - started) * 1000}
            with journal.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            startup["peak_inference_vram_gib"] = torch.cuda.max_memory_allocated() / 1024 ** 3
            startup["peak_process_ram_gib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 ** 2
            journal.with_name("server_resources.json").write_text(json.dumps(startup, ensure_ascii=False, indent=2) + "\n")
            return {"id": response_id, "object": "chat.completion", "created": int(time.time()),
                    "model": runtime["model"], "choices": [{"index": 0, "finish_reason": reason,
                    "message": {"role": "assistant", "content": tokenizer.decode(generated, skip_special_tokens=True)}}],
                    "usage": {"prompt_tokens": input_tokens, "completion_tokens": count,
                              "total_tokens": input_tokens + count}}

    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
