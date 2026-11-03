import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from common import MODES, ROOT, load_config, read_jsonl
from .evaluate import evaluate


COLUMNS = ["name", "answer_accuracy", "Q", "mean_input_tokens", "mean_output_tokens",
           "mean_llm_calls", "mean_cost", "median_latency_ms"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", required=True)
    parser.add_argument("--questions", required=True)
    parser.add_argument("--gold", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--endpoint-log")
    parser.add_argument("--output-dir", default=str(ROOT))
    parser.add_argument("--artifacts-dir", default=str(ROOT / "workspace" / "artifacts"))
    args = parser.parse_args()
    config, runtime = load_config(args.config)
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    artifacts = Path(args.artifacts_dir).resolve()
    artifacts.mkdir(parents=True, exist_ok=True)
    questions, gold = read_jsonl(args.questions), read_jsonl(args.gold)
    corpus = read_jsonl(Path(args.index) / "corpus.jsonl")
    summaries = []
    for mode in MODES:
        folder = artifacts / mode
        folder.mkdir(exist_ok=True)
        answers, trace = folder / "answers.jsonl", folder / "trace.jsonl"
        env = dict(os.environ)
        env["SUBMISSION_ID"] = os.environ["SUBMISSION_ID"] + ":" + mode
        subprocess.run([sys.executable, str(ROOT / "predict.py"), "--index", str(Path(args.index).resolve()),
                        "--questions", str(Path(args.questions).resolve()), "--config", str(Path(args.config).resolve()),
                        "--output", str(answers), "--trace", str(trace), "--mode", mode],
                       env=env, check=True)
        records = {r["response_id"]: r for r in read_jsonl(args.endpoint_log)} if args.endpoint_log else None
        summary = evaluate(questions, gold, read_jsonl(answers), read_jsonl(trace),
                           {b["block_id"] for b in corpus}, runtime, mode, records)
        (folder / "metrics.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        summaries.append(summary)
        print(f'{mode}: accuracy={summary["answer_accuracy"]:.3f}, Q={summary["Q"]}, accounted={summary["efficiency_eligible"]}', flush=True)
    with (output / "experiments.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(summaries)
    for source, target in (("answers", "dev_answers"), ("trace", "dev_trace")):
        shutil.copyfile(artifacts / "final" / (source + ".jsonl"), output / (target + ".jsonl"))
    metadata = {"config": config, "runtime": runtime}
    (artifacts / "run_manifest.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    subprocess.run([sys.executable, "-m", "support.make_report", "--results", str(output),
                    "--questions", str(Path(args.questions).resolve()), "--gold", str(Path(args.gold).resolve()),
                    "--index", str(Path(args.index).resolve()), "--artifacts-dir", str(artifacts)],
                   cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
