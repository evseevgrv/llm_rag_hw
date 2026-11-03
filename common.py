import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
MODES = ("no_context", "baseline", "final", "no_dense", "no_neighbors")
CATEGORIES = ("conditions", "comparison", "calculation", "insufficient")


def read_jsonl(path):
    with open(path, encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_config(path):
    path = Path(path).resolve()
    config = json.loads(path.read_text(encoding="utf-8"))
    runtime = json.loads((path.parent / config["runtime"]).read_text(encoding="utf-8"))
    return config, runtime


def answer_errors(answer, question, block_ids, no_context=False):
    errors = []
    keys = {"id", "answer_ids", "abstain", "rationale", "citations"}
    if not isinstance(answer, dict) or set(answer) != keys:
        return ["answer fields"]
    if answer["id"] != question["id"]:
        errors.append("question id")
    if type(answer["abstain"]) is not bool:
        errors.append("abstain must be boolean")
    rationale = answer["rationale"]
    if not isinstance(rationale, str) or not rationale.strip() or len(rationale) > 1200:
        errors.append("rationale length/type")
    for name, allowed in (("answer_ids", {o["id"] for o in question["options"]}),
                          ("citations", set(block_ids))):
        values = answer[name]
        if not isinstance(values, list) or any(not isinstance(x, str) for x in values):
            errors.append(name + " type")
        elif len(set(values)) != len(values) or not set(values) <= allowed:
            errors.append(name + " unknown/duplicate")
    if errors:
        return errors
    if len(answer["citations"]) > 4:
        errors.append("too many citations")
    if answer["abstain"]:
        if answer["answer_ids"] or answer["citations"]:
            errors.append("nonempty refusal")
    elif not answer["answer_ids"] or (not no_context and not answer["citations"]):
        errors.append("empty answer/evidence")
    if no_context and answer["citations"]:
        errors.append("citations without context")
    return errors
