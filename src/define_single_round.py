#!/usr/bin/env python3
"""
Step 5: Decide your "single round" definition (decode step vs context length)

Given your current engine (gpt2-medium-with-past ONNX/TRT), the fastest + simplest “single round”
that matches your engine inputs is:

  Single round = ONE decode step with past (input_ids length = 1) at a chosen context length L

Where:
  - past_key_values.* have sequence length = (L - 1)
  - attention_mask length = L
  - position_ids length = 1 (usually position = L-1)
  - input_ids length = 1 (the next token to decode)

This script creates a reusable “plan” file that lists the buckets (L values) you will measure,
and how to map L -> input shapes (past_len, mask_len, pos).

Outputs:
  single_round_plan.json

Usage:
  python3 step5_define_single_round.py --out single_round_plan.json
  python3 step5_define_single_round.py --contexts 16 32 64 128 256 512 --out plan.json

You’ll use this plan in Step 6 (measurement harness).
"""

import argparse
import json
from dataclasses import dataclass, asdict
from typing import List


@dataclass
class SingleRoundPlan:
    definition: str
    model_inputs: dict
    contexts: List[int]
    notes: List[str]


def build_plan(contexts: List[int]) -> SingleRoundPlan:
    # Ensure valid GPT-2 max context (typically 1024), but your trtexec maxShapes used 512.
    # We'll stick to <=512 because that's what you built in the engine command.
    contexts = sorted(set(int(c) for c in contexts))
    for c in contexts:
        if c < 2:
            raise ValueError("Context length must be >= 2 (because past_len = L-1).")
        if c > 512:
            raise ValueError("Context length > 512 not supported by your current TRT profile (maxShapes attention_mask=1x512).")

    return SingleRoundPlan(
        definition="ONE decode step with KV-cache (input_ids=1 token) at context length L",
        model_inputs={
            "input_ids": "shape = [B, 1] (B=1)  # one token decode",
            "position_ids": "shape = [B, 1], value ~= L-1",
            "attention_mask": "shape = [B, L]",
            "past_key_values.*.key": "shape = [B, NHEAD, L-1, HDIM]",
            "past_key_values.*.value": "shape = [B, NHEAD, L-1, HDIM]",
            "outputs": {
                "logits": "shape = [B, 1, vocab]",
                "present.*": "shape = [B, NHEAD, L, HDIM]  # appends one token to cache",
            },
        },
        contexts=contexts,
        notes=[
            "This matches your exported ONNX inputs (input_ids, position_ids, attention_mask, past_key_values.*).",
            "For each prompt of token length T, you can set L = min(T, 512) for measuring one decode step at that context.",
            "Step 6 will allocate device buffers for MAX shapes and then run enqueue with the chosen L (dynamic shapes).",
            "If you want to measure 'prefill' (processing the full prompt) you'll need a different engine/profile (input_ids length = L).",
        ],
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--contexts",
        nargs="+",
        type=int,
        default=[16, 32, 64, 128, 256, 512],
        help="Context-length buckets (L) to measure. Must be <=512 for your current plan.",
    )
    ap.add_argument("--out", default="single_round_plan.json")
    args = ap.parse_args()

    plan = build_plan(args.contexts)

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(asdict(plan), f, indent=2)

    print(f"Wrote single-round plan to: {args.out}")
    print("\nPlan summary:")
    print(f"  Definition: {plan.definition}")
    print(f"  Context buckets: {plan.contexts}")
    print("  Next: Step 6 uses this file to run warmup + repeated timing per bucket.")


if __name__ == "__main__":
    main()
