"""
Inventory AI Decoder + Deployment Lab
=====================================

This module is intentionally lightweight and dependency-free.  It turns the live
SQLite inventory rows and RAG retrieval results into classroom-style LLM decoding
and deployment demonstrations inspired by the user's lecture prototype files:

greedy decoding, beam search, temperature/top-k/top-p sampling, repetition
penalty, KV-cache cost intuition, continuous batching, quantization, and a small
model-cascade decision.

The production /ai page can import this file safely on Railway because it uses
only Python's standard library.  The heavier PyTorch decoder remains in
app/ai_transformer_decoder.py for offline experiments.
"""

from __future__ import annotations

import math
import random
import re
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Sequence, Tuple


LAB_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "do", "for", "from",
    "have", "i", "in", "is", "it", "me", "my", "of", "on", "or", "the",
    "to", "we", "what", "where", "which", "with", "you", "any", "all"
}


def clean_tokens(text: str) -> List[str]:
    words = re.findall(r"[a-zA-Z0-9_./+-]+", (text or "").lower())
    return [w for w in words if len(w) > 1 and w not in LAB_STOPWORDS]


def item_to_sentence(item) -> str:
    tags = item["tags"] or ""
    notes = item["notes"] or ""
    return (
        f"item {item['id']} {item['value_model']} type {item['item_type']} "
        f"quantity {item['quantity']} location {item['location']} tags {tags} notes {notes}"
    )


def build_bigram_model(texts: Iterable[str]) -> Dict[str, Counter]:
    """A tiny transparent language model used only to demonstrate decoding rules."""
    model: Dict[str, Counter] = defaultdict(Counter)
    for text in texts:
        tokens = ["<bos>"] + clean_tokens(text) + ["<eos>"]
        for left, right in zip(tokens, tokens[1:]):
            model[left][right] += 1
    return model


def next_distribution(model: Dict[str, Counter], previous: str) -> List[Tuple[str, float]]:
    counts = model.get(previous) or model.get("<bos>") or Counter({"<eos>": 1})
    total = sum(counts.values()) or 1
    dist = [(tok, count / total) for tok, count in counts.items()]
    return sorted(dist, key=lambda x: x[1], reverse=True)


def greedy_decode(model: Dict[str, Counter], max_tokens: int = 16) -> str:
    out = []
    prev = "<bos>"
    for _ in range(max_tokens):
        tok, _ = next_distribution(model, prev)[0]
        if tok == "<eos>":
            break
        out.append(tok)
        prev = tok
    return " ".join(out)


def beam_search_decode(model: Dict[str, Counter], beam_width: int = 3, max_tokens: int = 16) -> str:
    beams: List[Tuple[List[str], str, float]] = [([], "<bos>", 0.0)]
    for _ in range(max_tokens):
        candidates: List[Tuple[List[str], str, float]] = []
        for seq, prev, score in beams:
            for tok, p in next_distribution(model, prev)[:beam_width]:
                new_seq = seq if tok == "<eos>" else seq + [tok]
                candidates.append((new_seq, tok, score + math.log(max(p, 1e-12))))
        candidates.sort(key=lambda x: x[2] / max(1, len(x[0])), reverse=True)
        beams = candidates[:beam_width]
        if beams and beams[0][1] == "<eos>":
            break
    return " ".join(beams[0][0]) if beams else ""


def sample_decode(
    model: Dict[str, Counter],
    max_tokens: int = 16,
    temperature: float = 0.8,
    top_k: int = 5,
    top_p: float = 0.9,
    repetition_penalty: float = 1.15,
    seed: int = 42,
) -> str:
    rng = random.Random(seed)
    out: List[str] = []
    prev = "<bos>"
    used = Counter()
    for _ in range(max_tokens):
        dist = next_distribution(model, prev)
        adjusted = []
        for tok, p in dist:
            # Repetition penalty lowers probability of already used words.
            p = p / (repetition_penalty ** used[tok])
            # Temperature controls sharpness. Lower = safer, higher = more diverse.
            p = max(p, 1e-12) ** (1.0 / max(temperature, 1e-6))
            adjusted.append((tok, p))
        adjusted.sort(key=lambda x: x[1], reverse=True)
        adjusted = adjusted[:max(1, top_k)]
        total = sum(p for _, p in adjusted) or 1.0
        adjusted = [(tok, p / total) for tok, p in adjusted]
        # nucleus / top-p truncation
        nucleus = []
        cumulative = 0.0
        for tok, p in adjusted:
            nucleus.append((tok, p))
            cumulative += p
            if cumulative >= top_p:
                break
        total = sum(p for _, p in nucleus) or 1.0
        r = rng.random()
        cumulative = 0.0
        chosen = nucleus[-1][0]
        for tok, p in nucleus:
            cumulative += p / total
            if r <= cumulative:
                chosen = tok
                break
        if chosen == "<eos>":
            break
        out.append(chosen)
        used[chosen] += 1
        prev = chosen
    return " ".join(out)


def kv_cache_cost(seq_len: int, new_tokens: int) -> Dict[str, int | float]:
    """Simple attention-cost intuition: full recompute vs incremental KV cache."""
    full = sum((seq_len + i) ** 2 for i in range(1, new_tokens + 1))
    cached = seq_len * seq_len + sum(seq_len + i for i in range(1, new_tokens + 1))
    speedup = round(full / max(cached, 1), 2)
    return {"full_recompute_units": full, "kv_cache_units": cached, "estimated_speedup": speedup}


def continuous_batching_demo(num_requests: int, avg_prompt_tokens: int, avg_new_tokens: int) -> Dict[str, int | str]:
    static_padding_cost = num_requests * (avg_prompt_tokens + avg_new_tokens) * num_requests
    continuous_cost = num_requests * avg_prompt_tokens + num_requests * avg_new_tokens
    return {
        "requests": num_requests,
        "static_batch_cost_units": static_padding_cost,
        "continuous_batch_cost_units": continuous_cost,
        "idea": "Serve many short inventory questions by filling freed slots instead of waiting for the slowest request.",
    }


def quantization_estimate(param_count: int = 120_000) -> Dict[str, str]:
    fp32_mb = param_count * 4 / (1024 * 1024)
    int8_mb = param_count * 1 / (1024 * 1024)
    int4_mb = param_count * 0.5 / (1024 * 1024)
    return {
        "fp32": f"{fp32_mb:.2f} MB",
        "int8": f"{int8_mb:.2f} MB",
        "int4": f"{int4_mb:.2f} MB",
        "deployment_note": "For Railway, keep the live system RAG-first; train or quantize the decoder offline.",
    }


def cascade_decision(question: str, retrieved_count: int, max_score: float) -> Dict[str, str]:
    q = (question or "").strip()
    if not q:
        route = "help-mode"
        reason = "No user question yet."
    elif retrieved_count and max_score >= 0.35:
        route = "fast-local-rag"
        reason = "High retrieval confidence; answer from database evidence."
    elif retrieved_count:
        route = "rag-plus-decoder-explanation"
        reason = "Some evidence exists, but confidence is moderate; show evidence and avoid overclaiming."
    else:
        route = "safe-fallback"
        reason = "No grounded evidence; refuse to invent inventory facts."
    return {"route": route, "reason": reason}


def build_ai_lab_summary(question: str, result: Dict, items: Sequence) -> Dict:
    retrieved = result.get("retrieved", []) if result else []
    evidence_items = [r["item"] for r in retrieved] or list(items)[:8]
    texts = [item_to_sentence(item) for item in evidence_items]
    if not texts:
        texts = ["empty inventory no grounded answer available"]

    model = build_bigram_model(texts)
    max_score = max([float(r.get("score", 0.0)) for r in retrieved] or [0.0])
    seq_len = max(8, len(clean_tokens(question)) + 8)

    return {
        "decoding": {
            "greedy": greedy_decode(model),
            "beam_search": beam_search_decode(model, beam_width=3),
            "temperature_topk_topp": sample_decode(model, temperature=0.9, top_k=5, top_p=0.9),
        },
        "deployment": {
            "kv_cache": kv_cache_cost(seq_len=seq_len, new_tokens=24),
            "continuous_batching": continuous_batching_demo(num_requests=6, avg_prompt_tokens=seq_len, avg_new_tokens=24),
            "quantization": quantization_estimate(),
            "cascade": cascade_decision(question, len(retrieved), max_score),
        },
        "evaluation_hooks": [
            "Retrieval precision@k: check whether the correct item appears in the retrieved evidence.",
            "Hallucination check: answer must cite inventory rows or explicitly say no match.",
            "Ablation 1: TF-IDF retrieval only vs retrieval + synonym expansion.",
            "Ablation 2: greedy decoder vs sampling decoder for explanation diversity.",
            "Error taxonomy: no-match, wrong-type match, plural/singular failure, low-stock false positive.",
        ],
    }
