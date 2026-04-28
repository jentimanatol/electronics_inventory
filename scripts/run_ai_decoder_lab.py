#!/usr/bin/env python3
"""Run the dependency-free decoder/deployment lab from a SQLite inventory DB."""

import argparse
import json
import sqlite3
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.ai_decoder_lab import build_ai_lab_summary


def load_items(db_path):
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        return conn.execute("SELECT * FROM items ORDER BY updated_at DESC, id DESC").fetchall()
    finally:
        conn.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="app/uploads/inventory.db")
    parser.add_argument("--question", default="do we have any sensors?")
    args = parser.parse_args()

    items = load_items(args.db)
    fake_result = {"retrieved": [{"item": item, "score": 1.0} for item in items[:5]]}
    lab = build_ai_lab_summary(args.question, fake_result, items)
    print(json.dumps(lab, indent=2))


if __name__ == "__main__":
    main()
