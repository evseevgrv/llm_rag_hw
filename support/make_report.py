import argparse
import json
from pathlib import Path

from common import MODES, ROOT, read_jsonl


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True)
    parser.add_argument("--questions", required=True)
    parser.add_argument("--gold", required=True)
    parser.add_argument("--index", required=True)
    parser.add_argument("--artifacts-dir", default=str(ROOT / "workspace" / "artifacts"))
    args = parser.parse_args()
    output = Path(args.results)
    artifacts = Path(args.artifacts_dir)
    summaries = {m: json.loads((artifacts / m / "metrics.json").read_text()) for m in MODES}
    questions = {q["id"]: q for q in read_jsonl(args.questions)}
    gold = {g["id"]: g for g in read_jsonl(args.gold)}
    blocks = {b["block_id"]: b for b in read_jsonl(Path(args.index) / "corpus.jsonl")}
    manifest = json.loads((Path(args.index) / "manifest.json").read_text())
    text = (Path(__file__).resolve().parent / "report_base.md").read_text(encoding="utf-8")
    text += "\n## Эксперименты\n\n| Конфигурация | Accuracy | Q | Вход | Выход | Вызовы | C | Медиана, мс |\n|---|---:|---:|---:|---:|---:|---:|---:|\n"
    for mode, result in summaries.items():
        fields = [mode] + ["—" if result[k] is None else f"{result[k]:.3f}" for k in
                          ("answer_accuracy", "Q", "mean_input_tokens", "mean_output_tokens", "mean_llm_calls", "mean_cost", "median_latency_ms")]
        text += "| " + " | ".join(fields) + " |\n"
    text += "\nРезультаты получены на 15 открытых вопросах.\n"
    for mode, component in (("no_dense", "dense-поиска"), ("no_neighbors", "соседних фрагментов")):
        delta = summaries["final"]["Q"] - summaries[mode]["Q"]
        conclusion = "повысило" if delta > 0 else "снизило" if delta < 0 else "не изменило"
        text += f"\nДобавление {component} {conclusion} Q: Δ={delta:+.4f} относительно соответствующей абляции. Размер набора не позволяет судить о статистической значимости разницы.\n"
    baseline = summaries["baseline"]["mean_cost"]
    final = summaries["final"]
    efficiency = min(1, baseline / max(final["mean_cost"], 1)) if baseline is not None and final["mean_cost"] is not None and final["efficiency_eligible"] else 0
    score = 75 * final["Q"] + 25 * final["Q"] * efficiency
    text += f"\nЛокальный C_ref — стоимость зафиксированного BM25-baseline. E={efficiency:.4f}, учебный итог={score:.3f}. Это не балл проверяющей системы. Проверяемый расход итогового прогона: {final['efficiency_eligible']}.\n"
    text += "\n## Разбор ошибок\n"
    count = 0
    for mode in ("final", "baseline", "no_dense", "no_neighbors", "no_context"):
        answers = {a["id"]: a for a in read_jsonl(artifacts / mode / "answers.jsonl")}
        traces = {t["id"]: t for t in read_jsonl(artifacts / mode / "trace.jsonl")}
        for result in summaries[mode]["details"]:
            if result["score"] == 1 or count == 3:
                continue
            qid = result["id"]
            answer, trace = answers[qid], traces[qid]
            sources = set(trace["context_block_ids"])
            complete = any(set(s) <= sources for s in gold[qid]["citation_sets"])
            if result["errors"]:
                cause = "Нарушение протокола/техническая ошибка: " + ", ".join(result["errors"])
            elif not result["exact"]:
                cause = ("Неверный вывод при наличии одного из эталонных наборов блоков в контексте."
                         if complete else "В контексте нет полного эталонного набора, требуется проверить полноту поиска и прочитать найденные фрагменты.")
                if gold[qid]["abstain"]:
                    cause = "Модель выбрала значение при недостаточных данных вопроса: " + gold[qid]["missing_information"]
            else:
                cause = "Набор вариантов верный, но ссылки отличаются от допустимого эталонного набора."
            count += 1
            text += f"\n{count}. {qid} ({mode}). {questions[qid]['question']}\n\n"
            text += f"Ответ: {answer['answer_ids']}, отказ={answer['abstain']}, эталон: {gold[qid]['answer_ids']}, отказ={gold[qid]['abstain']}. {cause}\n\n"
            text += "Найденные блоки: " + ", ".join(trace["retrieved_block_ids"]) + ".\n\n"
            for bid in trace["context_block_ids"][:2]:
                excerpt = blocks[bid]["text"][:250].replace("\n", " ")
                text += f"{bid}: «{excerpt}…»\n\n"
        if count == 3:
            break
    if count < 3:
        text += f"\nВо всех пяти прогонах найдено {count} ошибок, недостающие примеры не выдумывались.\n"
    text += (f"\n## Ресурсы\n\nИндекс: {manifest['blocks']} блоков, {manifest['chunks']} фрагментов, "
             f"сборка {manifest['build_seconds']:.2f} с, пик RAM процесса {manifest['peak_ram_gib']:.3f} ГиБ, "
             f"размер {manifest['index_bytes'] / 1024**2:.2f} МиБ, включая локальные веса embedding-модели и токенизаторы. "
             f"Оборудование: {manifest['hardware']}. Загрузка весов и прогрев отдельно от latency вопросов.\n")
    server = artifacts / "server_resources.json"
    if server.exists():
        resource = json.loads(server.read_text())
        text += f"\nGPU: {resource['gpu']}, загрузка {resource['load_seconds']:.2f} с, прогрев {resource['warmup_seconds']:.2f} с, пик VRAM при загрузке/прогреве {resource['peak_vram_gib']:.3f} ГиБ.\n"
        if "peak_inference_vram_gib" in resource:
            text += f"Пик VRAM с учетом инференса: {resource['peak_inference_vram_gib']:.3f} ГиБ, пик RAM серверного процесса: {resource['peak_process_ram_gib']:.3f} ГиБ.\n"
    prediction_resources = artifacts / "final" / "trace.startup.json"
    if prediction_resources.exists():
        resource = json.loads(prediction_resources.read_text())
        text += f"\nПик RAM процесса поиска и предсказания: {resource['peak_process_ram_gib']:.3f} ГиБ, подготовка {resource['startup_ms']:.1f} мс, включая CPU-прогрев {resource['embedding_warmup_ms']:.1f} мс.\n"
    text += "\nОбоснования и причины ошибок нужно проверить вручную. Автоматическая оценка не проверяет математическую истинность ответа.\n"
    if len(text.split()) > 1500:
        raise ValueError("Report exceeds 1500 whitespace-separated words, shorten excerpts")
    (output / "report.md").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
