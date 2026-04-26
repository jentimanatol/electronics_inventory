"""
Tiny Inventory Transformer Decoder Prototype
==========================================

Purpose
-------
This file is intentionally separate from the production FastAPI app.
The live Railway app can keep using the fast local RAG assistant, while this
file gives the class project a real transformer-decoder component that can be
trained/evaluated offline on inventory text exported from SQLite.

Why separate?
-------------
Railway free/small deployments should not be forced to install PyTorch or load a
model at web-start. This module imports torch only when executed, so the website
continues to run even when torch is not installed.

Example commands
----------------
Install optional AI dependencies locally:
    pip install -r requirements-ai.txt

Train from the Railway/backup SQLite database:
    python app/ai_transformer_decoder.py --db uploads/inventory.db --epochs 20 --out results/inventory_decoder.pt

Generate a short inventory-style completion:
    python app/ai_transformer_decoder.py --db uploads/inventory.db --load results/inventory_decoder.pt --prompt "project: smart garden parts:" --generate

Model type
----------
Decoder-only Transformer, GPT-style next-character prediction:
    tokens -> embedding + positional embedding -> causal self-attention blocks -> logits

This mirrors the course examples: self-attention, multi-head attention, and
transformer stacks, but scaled down so it can run on CPU with a tiny private
inventory dataset.
"""

from __future__ import annotations

import argparse
import math
import os
import random
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence


@dataclass
class DecoderConfig:
    block_size: int = 128
    n_embd: int = 96
    n_head: int = 4
    n_layer: int = 2
    dropout: float = 0.1
    batch_size: int = 32
    learning_rate: float = 3e-4
    epochs: int = 10
    seed: int = 42


def set_seed(seed: int) -> None:
    random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def load_inventory_rows(db_path: str | os.PathLike[str]) -> List[sqlite3.Row]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            """
            SELECT id, category, item_type, value_model, normalized_value,
                   quantity, location, tags, notes, structured_name
            FROM items
            ORDER BY id ASC
            """
        ).fetchall()
    finally:
        conn.close()
    return rows


def row_to_training_line(row: sqlite3.Row) -> str:
    tags = row["tags"] or "no tags"
    notes = row["notes"] or "no notes"
    return (
        f"item #{row['id']}: {row['value_model']} | type: {row['item_type']} | "
        f"category: {row['category']} | qty: {row['quantity']} | "
        f"location: {row['location']} | tags: {tags} | notes: {notes}\n"
    )


def build_inventory_corpus(db_path: str | os.PathLike[str]) -> str:
    rows = load_inventory_rows(db_path)
    if not rows:
        return "empty inventory\n"

    lines = ["electronics inventory language model corpus\n"]
    lines.extend(row_to_training_line(row) for row in rows)
    lines.append("\nquestion: what should I restock?\nanswer: check items with quantity less than or equal to 2.\n")
    lines.append("question: do I have duplicates?\nanswer: compare category, item type, and normalized value.\n")
    lines.append("question: where is a part?\nanswer: search the matching item and report its location.\n")
    return "".join(lines)


class CharVocabulary:
    def __init__(self, text: str):
        chars = sorted(set(text))
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for ch, i in self.stoi.items()}

    def encode(self, text: str) -> List[int]:
        # Unknown characters are skipped instead of crashing during demos.
        return [self.stoi[ch] for ch in text if ch in self.stoi]

    def decode(self, ids: Sequence[int]) -> str:
        return "".join(self.itos[int(i)] for i in ids if int(i) in self.itos)

    def __len__(self) -> int:
        return len(self.stoi)


def require_torch():
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
    except ImportError as exc:
        raise SystemExit(
            "PyTorch is not installed. Run: pip install -r requirements-ai.txt\n"
            "The production FastAPI app does not need torch; only this optional decoder prototype does."
        ) from exc
    return torch, nn, F


def make_model_classes():
    torch, nn, F = require_torch()

    class CausalSelfAttention(nn.Module):
        def __init__(self, cfg: DecoderConfig):
            super().__init__()
            assert cfg.n_embd % cfg.n_head == 0
            self.n_head = cfg.n_head
            self.head_dim = cfg.n_embd // cfg.n_head
            self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd)
            self.proj = nn.Linear(cfg.n_embd, cfg.n_embd)
            self.dropout = nn.Dropout(cfg.dropout)
            mask = torch.tril(torch.ones(cfg.block_size, cfg.block_size)).view(1, 1, cfg.block_size, cfg.block_size)
            self.register_buffer("mask", mask)

        def forward(self, x):
            B, T, C = x.shape
            qkv = self.qkv(x)
            q, k, v = qkv.split(C, dim=2)
            q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
            k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
            v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

            att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
            att = F.softmax(att, dim=-1)
            att = self.dropout(att)
            y = att @ v
            y = y.transpose(1, 2).contiguous().view(B, T, C)
            return self.dropout(self.proj(y))

    class DecoderBlock(nn.Module):
        def __init__(self, cfg: DecoderConfig):
            super().__init__()
            self.ln1 = nn.LayerNorm(cfg.n_embd)
            self.attn = CausalSelfAttention(cfg)
            self.ln2 = nn.LayerNorm(cfg.n_embd)
            self.mlp = nn.Sequential(
                nn.Linear(cfg.n_embd, 4 * cfg.n_embd),
                nn.GELU(),
                nn.Linear(4 * cfg.n_embd, cfg.n_embd),
                nn.Dropout(cfg.dropout),
            )

        def forward(self, x):
            x = x + self.attn(self.ln1(x))
            x = x + self.mlp(self.ln2(x))
            return x

    class TinyInventoryDecoder(nn.Module):
        def __init__(self, vocab_size: int, cfg: DecoderConfig):
            super().__init__()
            self.cfg = cfg
            self.token_emb = nn.Embedding(vocab_size, cfg.n_embd)
            self.pos_emb = nn.Embedding(cfg.block_size, cfg.n_embd)
            self.blocks = nn.Sequential(*[DecoderBlock(cfg) for _ in range(cfg.n_layer)])
            self.ln_f = nn.LayerNorm(cfg.n_embd)
            self.head = nn.Linear(cfg.n_embd, vocab_size)

        def forward(self, idx, targets=None):
            B, T = idx.shape
            if T > self.cfg.block_size:
                idx = idx[:, -self.cfg.block_size :]
                T = idx.shape[1]
            pos = torch.arange(0, T, device=idx.device)
            x = self.token_emb(idx) + self.pos_emb(pos)[None, :, :]
            x = self.blocks(x)
            logits = self.head(self.ln_f(x))
            loss = None
            if targets is not None:
                targets = targets[:, -T:]
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.reshape(-1))
            return logits, loss

        @torch.no_grad()
        def generate(self, idx, max_new_tokens=160, temperature=0.9):
            for _ in range(max_new_tokens):
                idx_cond = idx[:, -self.cfg.block_size :]
                logits, _ = self(idx_cond)
                logits = logits[:, -1, :] / max(temperature, 1e-6)
                probs = F.softmax(logits, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1)
                idx = torch.cat((idx, next_id), dim=1)
            return idx

    return torch, TinyInventoryDecoder


def make_batches(encoded: Sequence[int], cfg: DecoderConfig, device):
    torch, _, _ = require_torch()
    data = torch.tensor(encoded, dtype=torch.long, device=device)
    max_start = max(1, len(data) - cfg.block_size - 1)
    while True:
        starts = torch.randint(0, max_start, (cfg.batch_size,), device=device)
        x = torch.stack([data[i : i + cfg.block_size] for i in starts])
        y = torch.stack([data[i + 1 : i + cfg.block_size + 1] for i in starts])
        yield x, y


def train_decoder(db_path: str, out_path: str, cfg: DecoderConfig) -> None:
    set_seed(cfg.seed)
    torch, TinyInventoryDecoder = make_model_classes()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    corpus = build_inventory_corpus(db_path)
    vocab = CharVocabulary(corpus)
    encoded = vocab.encode(corpus)
    if len(encoded) < cfg.block_size + 2:
        encoded = encoded * ((cfg.block_size + 2) // max(1, len(encoded)) + 1)

    model = TinyInventoryDecoder(len(vocab), cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)
    batches = make_batches(encoded, cfg, device)

    model.train()
    for step in range(cfg.epochs * 50):
        x, y = next(batches)
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step % 25 == 0:
            print(f"step={step:04d} loss={loss.item():.4f} device={device}")

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "vocab": vocab.stoi, "cfg": cfg.__dict__}, out_path)
    print(f"saved decoder checkpoint to {out_path}")


def generate_from_decoder(db_path: str, checkpoint_path: str, prompt: str) -> str:
    torch, TinyInventoryDecoder = make_model_classes()
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    cfg = DecoderConfig(**ckpt["cfg"])
    vocab = CharVocabulary(build_inventory_corpus(db_path))
    vocab.stoi = ckpt["vocab"]
    vocab.itos = {i: ch for ch, i in vocab.stoi.items()}
    model = TinyInventoryDecoder(len(vocab), cfg)
    model.load_state_dict(ckpt["model"])
    model.eval()
    idx = torch.tensor([vocab.encode(prompt)], dtype=torch.long)
    out = model.generate(idx, max_new_tokens=180)
    return vocab.decode(out[0].tolist())


def main() -> None:
    parser = argparse.ArgumentParser(description="Train or run a tiny decoder-only transformer on inventory text.")
    parser.add_argument("--db", default="/app/uploads/inventory.db", help="Path to inventory SQLite database")
    parser.add_argument("--out", default="results/inventory_decoder.pt", help="Checkpoint output path")
    parser.add_argument("--load", default="", help="Checkpoint path for generation")
    parser.add_argument("--prompt", default="question: what parts do I have for a smart garden?\nanswer:")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--generate", action="store_true")
    args = parser.parse_args()

    cfg = DecoderConfig(epochs=args.epochs)
    if args.generate:
        if not args.load:
            raise SystemExit("Use --load results/inventory_decoder.pt with --generate")
        print(generate_from_decoder(args.db, args.load, args.prompt))
    else:
        train_decoder(args.db, args.out, cfg)


if __name__ == "__main__":
    main()
