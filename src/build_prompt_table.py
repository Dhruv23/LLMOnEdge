#!/usr/bin/env python3
"""
Step 4: Build a "prompt table" from amazon_polarity
- Extract text (title + content)
- Compute GPT-2 token_len
- Save prompts.parquet (or prompts.jsonl)

Usage:
  python3 step4_build_prompt_table.py --out prompts.parquet --split train --take 200000
  python3 step4_build_prompt_table.py --out prompts.jsonl  --split train --take 50000

Notes:
- Token length is computed with the Hugging Face GPT-2 tokenizer (gpt2).
- amazon_polarity columns: ['label','title','content'] -> prompt := title + "\n\n" + content
- This script streams (optional) so you don’t have to load all 3.6M into RAM.
"""

import argparse
import json
import os
from typing import Dict, Any

from datasets import load_dataset
from transformers import AutoTokenizer

# Optional parquet support
def _has_pyarrow() -> bool:
    try:
        import pyarrow  # noqa: F401
        return True
    except Exception:
        return False


def build_prompt(row: Dict[str, Any]) -> str:
    title = (row.get("title") or "").strip()
    content = (row.get("content") or "").strip()
    if title and content:
        return f"{title}\n\n{content}"
    return title or content


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="amazon_polarity")
    ap.add_argument("--split", default="train")
    ap.add_argument("--cache_dir", default=None)
    ap.add_argument("--streaming", action="store_true", help="Stream dataset (recommended for huge splits)")
    ap.add_argument("--take", type=int, default=200000, help="How many examples to export")
    ap.add_argument("--out", default="prompts.parquet", help="prompts.parquet or prompts.jsonl")
    ap.add_argument("--max_chars", type=int, default=0, help="Optional: truncate prompt text to N chars (0 = no truncation)")
    ap.add_argument("--tokenizer", default="gpt2", help="Tokenizer name (gpt2 matches GPT-2 BPE)")
    args = ap.parse_args()

    out_ext = os.path.splitext(args.out)[1].lower()
    if out_ext not in [".parquet", ".jsonl"]:
        raise ValueError("Output must be .parquet or .jsonl")

    if out_ext == ".parquet" and not _has_pyarrow():
        raise RuntimeError("pyarrow not installed. Install it or use --out prompts.jsonl")

    print(f"Loading dataset={args.dataset} split={args.split} streaming={args.streaming}")
    ds = load_dataset(
        args.dataset,
        split=args.split,
        cache_dir=args.cache_dir,
        streaming=args.streaming,
    )

    print(f"Loading tokenizer={args.tokenizer}")
    tok = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)

    # We'll build rows as:
    # {id, label, title, content, prompt, token_len, char_len}
    rows = []
    n = 0

    if args.streaming:
        it = iter(ds)
        for ex in it:
            prompt = build_prompt(ex)
            if args.max_chars and len(prompt) > args.max_chars:
                prompt = prompt[: args.max_chars]

            # GPT-2 token length (no special tokens for plain text)
            token_len = len(tok.encode(prompt, add_special_tokens=False))

            rows.append(
                {
                    "id": n,
                    "label": int(ex["label"]),
                    "title": ex.get("title", ""),
                    "content": ex.get("content", ""),
                    "prompt": prompt,
                    "char_len": len(prompt),
                    "token_len": int(token_len),
                }
            )
            n += 1
            if n >= args.take:
                break
    else:
        # Non-streaming: select first N rows
        if args.take > 0:
            ds = ds.select(range(min(args.take, len(ds))))
        for ex in ds:
            prompt = build_prompt(ex)
            if args.max_chars and len(prompt) > args.max_chars:
                prompt = prompt[: args.max_chars]
            token_len = len(tok.encode(prompt, add_special_tokens=False))
            rows.append(
                {
                    "id": n,
                    "label": int(ex["label"]),
                    "title": ex.get("title", ""),
                    "content": ex.get("content", ""),
                    "prompt": prompt,
                    "char_len": len(prompt),
                    "token_len": int(token_len),
                }
            )
            n += 1

    print(f"Built {len(rows)} prompt rows.")

    if out_ext == ".jsonl":
        with open(args.out, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"Wrote JSONL: {args.out}")
    else:
        import pandas as pd

        df = pd.DataFrame(rows)
        df.to_parquet(args.out, index=False)
        print(f"Wrote Parquet: {args.out}")

    # Quick sanity stats
    token_lens = [r["token_len"] for r in rows]
    char_lens = [r["char_len"] for r in rows]
    print("\nSanity stats:")
    print(f"  token_len: min={min(token_lens)}  p50={sorted(token_lens)[len(token_lens)//2]}  max={max(token_lens)}")
    print(f"  char_len : min={min(char_lens)}   p50={sorted(char_lens)[len(char_lens)//2]}   max={max(char_lens)}")
    print("\nNext: use this prompts table to bucket by token_len and run timing measurements.")


if __name__ == "__main__":
    main()
