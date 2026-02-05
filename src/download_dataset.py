#!/usr/bin/env python3
"""
Step 3: Download + cache dataset via Hugging Face `load_dataset`

Usage:
  python3 step3_download_cache.py --dataset amazon_polarity --split train --take 20000

Notes:
- Hugging Face Datasets caches automatically.
- You can control cache locations with env vars:
    HF_HOME, HF_DATASETS_CACHE, TRANSFORMERS_CACHE
  or pass --cache_dir.
"""

import argparse
import os
from datasets import load_dataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="amazon_polarity", help="HF dataset name")
    ap.add_argument("--config", default=None, help="Optional dataset config name")
    ap.add_argument("--split", default="train", help="Split to download (train/test/validation)")
    ap.add_argument("--cache_dir", default=None, help="Optional cache directory (overrides default)")
    ap.add_argument("--take", type=int, default=0, help="Optionally materialize first N rows to force full download")
    args = ap.parse_args()

    # Print cache-related env vars (useful for debugging where files land)
    print("Cache env:")
    for k in ["HF_HOME", "HF_DATASETS_CACHE", "TRANSFORMERS_CACHE"]:
        print(f"  {k}={os.environ.get(k, '')}")

    print(f"\nLoading dataset={args.dataset} config={args.config} split={args.split}")
    ds = load_dataset(
        args.dataset,
        args.config,
        split=args.split,
        cache_dir=args.cache_dir,
    )

    # This confirms download + shows what you got
    print("\nLoaded successfully.")
    print("Columns:", ds.column_names)
    print("Num rows:", len(ds))
    print("First row:", ds[0])

    # Optional: force materialization of first N rows (sometimes helpful to ensure everything is pulled)
    if args.take and args.take > 0:
        n = min(args.take, len(ds))
        _ = ds.select(range(n))
        print(f"\nMaterialized first {n} rows (select) to ensure cache population.")

    # Tell user where dataset cache likely is
    # (datasets decides exact subpaths; this is the root)
    print("\nCache location hints:")
    print("  If you passed --cache_dir, it is under that directory.")
    print("  Otherwise, HF uses ~/.cache/huggingface/datasets by default (unless env vars override).")


if __name__ == "__main__":
    main()
