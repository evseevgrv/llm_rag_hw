import re
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from rank_bm25 import BM25Okapi
from transformers import AutoModel, AutoTokenizer

from common import read_jsonl


def normalize_math(text):
    commands = {"gamma": "γ", "Gamma": "Γ", "zeta": "ζ", "varphi": "φ", "phi": "φ",
                "pi": "π", "alpha": "α", "beta": "β", "theta": "θ", "partial": "∂",
                "infty": "∞", "int": "∫", "oint": "∮", "sum": "∑", "ne": "≠",
                "le": "⩽", "leq": "⩽", "ge": "⩾", "geq": "⩾", "sin": "sin", "cos": "cos"}
    text = re.sub(r"\\mathbb\s*\{?([CRZN])\}?", lambda m: dict(C="ℂ", R="ℝ", Z="ℤ", N="ℕ")[m[1]], text)
    text = re.sub(r"\\([A-Za-z]+)", lambda m: commands.get(m[1], m[1]), text)
    return text.replace("$", "")


def terms(text):
    text = normalize_math(text).lower().replace("ё", "е")
    stopwords = set("какие какой которые все только пусть известно верны следует можно ли и на из не да нет или при для что это как по если то чтобы выберите нужно утверждения чему".split())
    words = re.findall(r"[^\W_]+|[∂∫∮∑∞ℂℝ⩽⩾]", text)
    words = [w for w in words if w not in stopwords and
             ((len(w) > 1 and not w.isdigit()) or w in "∂∫∮∑∞ℂℝ⩽⩾πγζφ")]
    return words + ["ru:" + w[:5] for w in words if re.fullmatch("[а-я]{6,}", w)]


def encode(texts, tokenizer, model, batch_size=16):
    vectors = []
    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            batch = tokenizer(texts[start:start + batch_size], padding=True,
                              truncation=True, max_length=512, return_tensors="pt")
            hidden = model(**batch).last_hidden_state
            mask = batch["attention_mask"].unsqueeze(-1)
            pooled = (hidden * mask).sum(1) / mask.sum(1)
            vectors.append(torch.nn.functional.normalize(pooled, dim=1).numpy())
    return np.concatenate(vectors)


def make_chunks(corpus, tokenizer, config):
    chunks = []
    size = config["chunk_tokens"]
    overlap = config["chunk_overlap"]
    for block in corpus:
        encoded = tokenizer(block["text"], add_special_tokens=False,
                            return_offsets_mapping=True, verbose=False)
        offsets = encoded["offset_mapping"]
        for start in range(0, len(offsets), size - overlap):
            end = min(start + size, len(offsets))
            text = block["text"][offsets[start][0]:offsets[end - 1][1]]
            chunks.append({**block, "text": text, "chunk_id": len(chunks)})
            if end == len(offsets):
                break
    return chunks


class SearchIndex:
    def __init__(self, path, config, dense=True):
        self.path = Path(path)
        self.config = config
        self.blocks = read_jsonl(self.path / "corpus.jsonl")
        self.positions = {b["block_id"]: i for i, b in enumerate(self.blocks)}
        self.chunks = read_jsonl(self.path / "chunks.jsonl")
        self.bm25 = BM25Okapi([terms(c["section"] + " " + c["text"]) for c in self.chunks],
                             k1=config["bm25_k1"], b=config["bm25_b"])
        if dense:
            torch.set_num_threads(config["cpu_threads"])
            self.tokenizer = AutoTokenizer.from_pretrained(self.path / "embedding", local_files_only=True)
            self.model = AutoModel.from_pretrained(self.path / "embedding", local_files_only=True).cpu().eval()
            self.vectors = np.load(self.path / "vectors.npy", mmap_mode="r")
            warmup = time.perf_counter()
            encode(["query: прогрев"], self.tokenizer, self.model)
            self.warmup_ms = (time.perf_counter() - warmup) * 1000

    def search(self, question, dense):
        query = question["question"] + "\n" + "\n".join(o["text"] for o in question["options"])
        k = self.config["retrieval_k"]
        lexical = self.bm25.get_scores(terms(query))
        bm = [int(i) for i in np.argsort(-lexical, kind="stable")[:k] if lexical[i] > 0]
        if not dense:
            return [self.chunks[i] for i in bm]
        vector = encode(["query: " + normalize_math(query)], self.tokenizer, self.model)[0]
        semantic = np.argsort(-(self.vectors @ vector), kind="stable")[:k].tolist()
        fused = defaultdict(float)
        for ranking in (bm, semantic):
            for rank, i in enumerate(ranking, 1):
                fused[i] += 1 / (self.config["rrf_k"] + rank)
        return [self.chunks[i] for i in sorted(fused, key=lambda i: (-fused[i], i))]

    def context(self, ranked, tokenizer, neighbors):
        selected = []
        seen = set()
        budget = self.config["context_tokens"]
        seeds = []
        for chunk in ranked:
            if chunk["block_id"] not in {s["block_id"] for s in seeds}:
                seeds.append(chunk)
            if len(seeds) == self.config["context_seeds"]:
                break
        candidates = list(seeds)
        if neighbors:
            for seed in seeds:
                pos = self.positions[seed["block_id"]]
                for delta in range(-self.config["neighbor_radius"], self.config["neighbor_radius"] + 1):
                    j = pos + delta
                    if 0 <= j < len(self.blocks) and self.blocks[j]["section"] == seed["section"]:
                        candidates.append(self.blocks[j])
        for chunk in candidates:
            if chunk["block_id"] in seen:
                continue
            header = f'[{chunk["block_id"]}; PDF {chunk["page"]}; {chunk["section"]}]\n'
            ids = tokenizer.encode(chunk["text"], add_special_tokens=False)
            text = tokenizer.decode(ids[:900], skip_special_tokens=True)
            entry = header + text
            cost = len(tokenizer.encode(entry, add_special_tokens=False)) + 2
            if cost > budget:
                continue
            budget -= cost
            selected.append({**chunk, "text": text, "entry": entry})
            seen.add(chunk["block_id"])
        return selected
