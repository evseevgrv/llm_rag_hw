import argparse
import json
import statistics
from collections import Counter
from pathlib import Path

from common import CATEGORIES, answer_errors, read_jsonl


def trace_errors(trace, runtime, endpoint_records=None):
    if trace is None:
        return ["missing trace"], False
    errors = []
    verified = trace.get("usage_verified") is True
    calls = trace.get("llm_calls")
    requests = trace.get("requests", [])
    if not isinstance(requests, list) or any(not isinstance(r, dict) for r in requests):
        return ["invalid request accounting"], False
    if type(calls) is not int or not 0 <= calls <= 4 or calls != len(requests):
        errors.append("call count")
    for key, limit in (("input_tokens", 16384), ("output_tokens", 4096)):
        value = trace.get(key)
        if type(value) is not int or not 0 <= value <= limit:
            errors.append(key)
            verified = False
        elif requests and sum(r.get(key) or 0 for r in requests) != value:
            errors.append(key + " sum")
    for request in requests:
        inp, out, maximum = (request.get(k) for k in ("input_tokens", "output_tokens", "max_tokens"))
        if any(type(x) is not int or x < 0 for x in (inp, out, maximum)):
            verified = False
            errors.append("unaccounted request")
            continue
        if inp + maximum > 8192 or out > maximum or maximum != runtime["generation"]["max_tokens"]:
            errors.append("request limit")
        if endpoint_records is not None:
            record = endpoint_records.get(request.get("response_id"))
            if (not record or record.get("id") != trace["id"]
                    or record.get("input_tokens") != inp or record.get("output_tokens") != out
                    or record.get("submission_id") != trace.get("submission_id")):
                errors.append("endpoint journal mismatch")
                verified = False
    if not calls:
        verified = False
    if trace.get("status") != "ok":
        errors.append("technical failure")
    return errors, verified


def citation_f1(predicted, acceptable):
    a, b = set(predicted), set(acceptable)
    return 2 * len(a & b) / (len(a) + len(b)) if a or b else 1.0


def evaluate(questions, gold, answers, traces, blocks, runtime, mode, endpoint_records=None):
    qmap = {q["id"]: q for q in questions}
    gmap = {g["id"]: g for g in gold}
    if set(qmap) != set(gmap) or len(qmap) != len(questions) or len(gmap) != len(gold):
        raise ValueError("Question/gold IDs must match and be unique")
    if {g["category"] for g in gold} != set(CATEGORIES):
        raise ValueError("All four categories are required for macro-average Q")
    answers = [a if isinstance(a, dict) and isinstance(a.get("id"), str) else
               {"id": f"__invalid_answer_{i}"} for i, a in enumerate(answers)]
    traces = [t if isinstance(t, dict) and isinstance(t.get("id"), str) else
              {"id": f"__invalid_trace_{i}"} for i, t in enumerate(traces)]
    amap, tmap = {a["id"]: a for a in answers}, {t["id"]: t for t in traces}
    acount, tcount = Counter(a["id"] for a in answers), Counter(t["id"] for t in traces)
    extras = (set(amap) | set(tmap)) - set(qmap)
    categories = {c: [] for c in CATEGORIES}
    details = []
    all_verified = not extras
    for qid, question in qmap.items():
        target, answer, trace = gmap[qid], amap.get(qid), tmap.get(qid)
        errors = answer_errors(answer, question, blocks, mode == "no_context")
        if acount[qid] != 1 or tcount[qid] != 1:
            errors.append("missing/duplicate/extra records")
        accounting, verified = trace_errors(trace, runtime, endpoint_records)
        errors += accounting
        all_verified &= verified and not [e for e in accounting if e != "technical failure"]
        citations = answer.get("citations") if isinstance(answer, dict) else None
        if trace and isinstance(citations, list) and all(isinstance(c, str) for c in citations) and not set(citations) <= set(trace.get("context_block_ids", [])):
            errors.append("citation outside shown context")
        exact = bool(not errors and answer["abstain"] == target["abstain"]
                     and set(answer["answer_ids"]) == set(target["answer_ids"]))
        f1 = 0.0
        if exact and not target["abstain"] and mode != "no_context":
            f1 = max((citation_f1(answer["citations"], c) for c in target["citation_sets"]), default=0.0)
        score = float(exact) if target["abstain"] else float(exact) * (0.8 + 0.2 * f1)
        categories[target["category"]].append(score)
        details.append({"id": qid, "category": target["category"], "exact": exact,
                        "citation_f1": f1, "score": score, "errors": errors})
    complete_usage = all(type(tmap.get(qid, {}).get(k)) is int for qid in qmap
                         for k in ("input_tokens", "output_tokens", "llm_calls"))
    summary = {"name": mode, "answer_accuracy": statistics.mean(d["exact"] for d in details),
               "Q": None if mode == "no_context" else statistics.mean(statistics.mean(v) for v in categories.values() if v),
               "mean_input_tokens": None, "mean_output_tokens": None, "mean_llm_calls": None,
               "mean_cost": None, "median_latency_ms": None,
               "efficiency_eligible": bool(all_verified), "details": details,
               "category_scores": {k: statistics.mean(v) if v else None for k, v in categories.items()},
               "extra_record_ids": sorted(extras)}
    if complete_usage:
        for key in ("input_tokens", "output_tokens", "llm_calls"):
            summary["mean_" + key] = statistics.mean(tmap[qid][key] for qid in qmap)
        summary["mean_cost"] = statistics.mean(tmap[q]["input_tokens"] + 4 * tmap[q]["output_tokens"]
                                               + 256 * tmap[q]["llm_calls"] for q in qmap)
    if all(type(tmap.get(qid, {}).get("elapsed_ms")) in (int, float) for qid in qmap):
        summary["median_latency_ms"] = statistics.median(tmap[q]["elapsed_ms"] for q in qmap)
    return summary


def read_results(path):
    rows = []
    with open(path, encoding="utf-8") as stream:
        for line in stream:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                rows.append(None)
    return rows


def main():
    parser = argparse.ArgumentParser()
    for key in ("questions", "gold", "answers", "trace", "corpus", "runtime", "output"):
        parser.add_argument("--" + key, required=True)
    parser.add_argument("--mode", default="final")
    parser.add_argument("--endpoint-log")
    parser.add_argument("--reference-cost", type=float)
    args = parser.parse_args()
    records = None
    if args.endpoint_log:
        records = {r["response_id"]: r for r in read_jsonl(args.endpoint_log)}
    summary = evaluate(read_jsonl(args.questions), read_jsonl(args.gold), read_results(args.answers),
                       read_results(args.trace), {b["block_id"] for b in read_jsonl(args.corpus)},
                       json.loads(Path(args.runtime).read_text()), args.mode, records)
    if args.reference_cost is not None and summary["Q"] is not None:
        efficiency = min(1, args.reference_cost / max(summary["mean_cost"] or 1, 1)) if summary["efficiency_eligible"] else 0
        summary.update(E=efficiency, total_score=75 * summary["Q"] + 25 * summary["Q"] * efficiency)
    Path(args.output).write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k not in ("details", "category_scores")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
