import argparse
import re
from pathlib import Path

from common import ROOT, write_jsonl
from .own_tasks import own_tasks
from .pdf_extract import extract_pdf


def find_block(corpus, page, text):
    return next(b["block_id"] for b in corpus if b["page"] == page and text in b["text"])


DEV_SOURCES = {
    1: [(11, "Теорема 2.1"), (11, "Определение 2.2")],
    2: [(12, "Наглядный способ"), (12, "В этих терминах")],
    3: [(11, "Теорема 2.1"), (13, "Определение 2.3")],
    4: [(26, "Теорема 4.1"), (28, "неотрицательным"), (31, "Замечание 4.1")],
    5: [(30, "Теорема 4.4")],
    6: [(31, "Теорема 5.1"), (33, "Следствие 5.1")],
    7: [(20, "Следовательно, равенство (3.3)"), (34, "Теорема 5.3")],
    8: [(35, "Теорема 5.4"), (35, "Тогда f является")],
    9: [(11, "Теорема 2.1"), (13, "Определение 2.3"), (34, "Теорема 5.3"), (13, "Система уравнений")],
    10: [(36, "Теорема 6.1")],
    11: [(37, "Теорема 6.2")],
    12: [(21, "Тригонометрические функции"), (37, "К целым функциям"), (38, "Теорема 6.3")],
    13: [(37, "Следствие 6.1")],
    14: [(26, "Однако,"), (28, "неотрицательным")],
    15: [(20, "Следовательно, равенство (3.3)"), (36, "Теорема 6.1")],
}


def parse_qa(path, corpus):
    text = Path(path).read_text(encoding="utf-8")
    parts = re.split(r"^## (\d+)\. [^\n]+\n", text, flags=re.M)
    questions, gold = [], []
    categories = {"условия": "conditions", "сопоставление": "comparison",
                  "применение": "calculation", "недостаточность данных": "insufficient"}
    for i in range(1, len(parts), 2):
        number, body = int(parts[i]), parts[i + 1]
        qid = f"q_{number:03d}"
        question = re.search(r"\*\*Вопрос\.\*\*\s*(.*?)\n\s*- \*\*A", body, re.S).group(1).strip()
        options = [{"id": m[0], "text": m[1].strip()} for m in
                   re.findall(r"^- \*\*([A-Z])\.\*\* (.+)$", body, re.M)]
        answer = re.search(r"\*\*Ответ:\*\*\s*([^\n]+)", body).group(1).strip()
        rationale = re.search(r"\*\*Ответ:\*\*[^\n]+\n\s*(.*?)\n\*\*Источник:", body, re.S).group(1).strip()
        category = categories[re.search(r"\*\*Тип:\*\* ([^.]+)", body).group(1)]
        abstain = answer.startswith("Отказ")
        sources = [find_block(corpus, p, s) for p, s in DEV_SOURCES[number]]
        questions.append({"id": qid, "question": question, "options": options})
        gold.append({"id": qid, "category": category,
                     "answer_ids": [] if abstain else re.findall("[A-Z]", answer),
                     "abstain": abstain, "citation_sets": [] if abstain else [sources],
                     "rationale": rationale, "supporting_block_ids": sources,
                     **({"missing_information": rationale} if abstain else {})})
    return questions, gold


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", required=True)
    parser.add_argument("--qa", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--own-output", default=str(ROOT))
    args = parser.parse_args()
    output = Path(args.output)
    corpus = extract_pdf(args.pdf)
    questions, gold = parse_qa(args.qa, corpus)
    own_questions, own_gold = own_tasks(corpus, find_block)
    for name, rows in (("corpus", corpus), ("dev_questions", questions), ("dev_gold", gold),
                       ("own_questions", own_questions), ("own_gold", own_gold)):
        write_jsonl(output / (name + ".jsonl"), rows)
    write_jsonl(Path(args.own_output) / "own_questions.jsonl", own_questions)
    write_jsonl(Path(args.own_output) / "own_gold.jsonl", own_gold)
    print(f"Prepared {len(corpus)} blocks, {len(questions)} dev and {len(own_questions)} own questions")


if __name__ == "__main__":
    main()
