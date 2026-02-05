#!/usr/bin/env python3
"""
Plot suite for LLM single-round measurements.csv

Usage:
  python3 plot_measurements.py --csv measurements.csv --outdir plots
Optional:
  python3 plot_measurements.py --csv measurements.csv --outdir plots --group-cols dataset,model,engine --logy

What it generates (PNG + a compact HTML index):
  01_latency_vs_tokens.png
  02_latency_vs_tokens_box_by_bucket.png
  03_latency_distribution_by_bucket.png
  04_latency_variability_by_bucket.png
  05_compute_vs_e2e.png (if gpu_compute_ms exists)
  06_breakdown_stacked_by_bucket.png (if h2d_ms/d2h_ms exist)
  07_throughput_tokens_per_s_by_bucket.png
  08_latency_cv_vs_tokens.png
  09_latency_outliers_table.csv
  summary_stats_by_bucket.csv
  index.html

Assumptions:
  - measurements.csv has at least one of: token_len, ctx_len, context_len, prompt_tokens, n_tokens
  - and at least one latency column: e2e_ms, latency_ms, total_ms
  - optional columns: gpu_compute_ms, enqueue_ms, h2d_ms, d2h_ms, run_id, repeat, seed, bucket

This script tries to auto-detect column names; you can override with flags.

No seaborn; matplotlib only.
"""

import argparse
import os
import math
import textwrap
from pathlib import Path
from typing import Optional, List, Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ---------------------------- helpers ----------------------------
CAND_TOK_COLS = ["ctx_len", "past_len", "token_len", "tokens", "prompt_tokens", "context_len", "n_tokens", "num_tokens"]

CAND_LAT_COLS = ["latency_mean_ms", "e2e_ms", "latency_ms", "total_ms", "end_to_end_ms", "host_latency_ms"]

CAND_GPU_COLS = ["gpu_compute_mean_ms", "gpu_compute_ms", "gpu_ms", "gpu_latency_ms", "gpu_time_ms"]

CAND_H2D_COLS = ["h2d_mean_ms", "h2d_ms", "h2d_latency_ms"]

CAND_D2H_COLS = ["d2h_mean_ms", "d2h_ms", "d2h_latency_ms"]

CAND_ENQ_COLS = ["enqueue_mean_ms", "enqueue_ms", "enqueue_time_ms"]

CAND_BUCKET_COLS = ["ctx_len", "bucket", "len_bucket", "context_bucket", "token_bucket"]



def pick_first_existing(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def ensure_outdir(outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)


def savefig(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()


def add_title_subtitle(title: str, subtitle: str = "") -> None:
    plt.title(title)
    if subtitle:
        plt.gcf().text(0.5, 0.01, subtitle, ha="center", va="bottom", fontsize=9, color="gray")


def quantile_buckets(series: pd.Series, buckets: int = 8) -> pd.Series:
    # robust bucketing even if many duplicates
    try:
        return pd.qcut(series, q=buckets, duplicates="drop")
    except Exception:
        # fallback to fixed-width bins
        return pd.cut(series, bins=buckets)


def robust_outlier_mask(x: pd.Series, k: float = 3.5) -> pd.Series:
    """
    MAD-based outliers (robust z-score).
    Returns boolean mask where True = outlier
    """
    x = pd.to_numeric(x, errors="coerce")
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med))
    if mad == 0 or np.isnan(mad):
        return pd.Series([False] * len(x), index=x.index)
    robust_z = 0.6745 * (x - med) / mad
    return np.abs(robust_z) > k


def group_key(df: pd.DataFrame, group_cols: List[str]) -> pd.Series:
    if not group_cols:
        return pd.Series(["all"] * len(df), index=df.index)
    for col in group_cols:
        if col not in df.columns:
            raise ValueError(f"--group-cols includes '{col}', but it is not in CSV columns: {list(df.columns)}")
    return df[group_cols].astype(str).agg(" | ".join, axis=1)


def maybe_log_scale(axis: str, enabled: bool) -> None:
    if enabled:
        if axis == "y":
            plt.yscale("log")
        elif axis == "x":
            plt.xscale("log")


# ---------------------------- plots ----------------------------

def plot_scatter_latency_vs_tokens(df: pd.DataFrame, tok_col: str, lat_col: str,
                                   outdir: Path, logy: bool, group_cols: List[str]) -> Path:
    path = outdir / "01_latency_vs_tokens.png"
    plt.figure(figsize=(9, 6))

    gk = group_key(df, group_cols)
    for key in sorted(gk.unique()):
        sub = df[gk == key]
        plt.scatter(sub[tok_col], sub[lat_col], s=10, alpha=0.5, label=key)

        # trend line (median per token)
        med = sub.groupby(tok_col)[lat_col].median().reset_index().sort_values(tok_col)
        if len(med) >= 2:
            plt.plot(med[tok_col], med[lat_col], linewidth=2)

    plt.xlabel(tok_col)
    plt.ylabel(f"{lat_col} (ms)")
    maybe_log_scale("y", logy)
    plt.legend(fontsize=8, loc="best")
    add_title_subtitle("End-to-end latency vs token/context length",
                       "Dots = individual runs; line = median latency per length")
    savefig(path)
    return path


def plot_box_latency_by_bucket(df: pd.DataFrame, tok_col: str, lat_col: str,
                               bucket_col: str, outdir: Path, logy: bool) -> Path:
    path = outdir / "02_latency_vs_tokens_box_by_bucket.png"
    plt.figure(figsize=(11, 6))

    # order buckets by min token value
    bucket_order = (
        df.groupby(bucket_col)[tok_col].min()
        .sort_values()
        .index.tolist()
    )
    data = [df.loc[df[bucket_col] == b, lat_col].dropna().values for b in bucket_order]

    plt.boxplot(data, labels=[str(b) for b in bucket_order], showfliers=False)
    plt.xticks(rotation=30, ha="right")
    plt.xlabel("Length bucket")
    plt.ylabel(f"{lat_col} (ms)")
    maybe_log_scale("y", logy)
    add_title_subtitle("Latency distribution by length bucket",
                       "Box = IQR; whiskers = 1.5×IQR; outliers hidden (see outlier table)")
    savefig(path)
    return path


def plot_hist_latency_by_bucket(df: pd.DataFrame, tok_col: str, lat_col: str,
                                bucket_col: str, outdir: Path, max_buckets: int = 8) -> Path:
    path = outdir / "03_latency_distribution_by_bucket.png"
    # pick up to max_buckets buckets evenly across token range
    ordered = (
        df.groupby(bucket_col)[tok_col].median()
        .sort_values()
        .index.tolist()
    )
    if len(ordered) > max_buckets:
        idx = np.linspace(0, len(ordered) - 1, max_buckets).round().astype(int)
        chosen = [ordered[i] for i in idx]
    else:
        chosen = ordered

    plt.figure(figsize=(11, 6))
    for b in chosen:
        sub = df[df[bucket_col] == b][lat_col].dropna()
        if len(sub) < 5:
            continue
        plt.hist(sub.values, bins=30, alpha=0.35, label=str(b), density=True)

    plt.xlabel(f"{lat_col} (ms)")
    plt.ylabel("Density")
    plt.legend(fontsize=8, loc="best")
    add_title_subtitle("Latency histograms for representative length buckets",
                       "Overlaid densities highlight variance/tails (paging/thrashing shows up as heavy tails)")
    savefig(path)
    return path


def plot_variability_by_bucket(df: pd.DataFrame, tok_col: str, lat_col: str,
                               bucket_col: str, outdir: Path) -> Path:
    path = outdir / "04_latency_variability_by_bucket.png"

    agg = df.groupby(bucket_col).agg(
        n=(lat_col, "count"),
        median_ms=(lat_col, "median"),
        p95_ms=(lat_col, lambda x: np.nanpercentile(x, 95)),
        p99_ms=(lat_col, lambda x: np.nanpercentile(x, 99)),
        std_ms=(lat_col, "std"),
        mean_ms=(lat_col, "mean"),
    ).reset_index()

    agg["cv"] = agg["std_ms"] / agg["mean_ms"]

    # sort by median token length
    order = df.groupby(bucket_col)[tok_col].median().sort_values().index
    agg = agg.set_index(bucket_col).loc[order].reset_index()

    plt.figure(figsize=(11, 6))
    x = np.arange(len(agg))
    plt.plot(x, agg["median_ms"], marker="o", label="median")
    plt.plot(x, agg["p95_ms"], marker="o", label="p95")
    plt.plot(x, agg["p99_ms"], marker="o", label="p99")
    plt.xticks(x, [str(b) for b in agg[bucket_col]], rotation=30, ha="right")
    plt.xlabel("Length bucket")
    plt.ylabel(f"{lat_col} (ms)")
    plt.legend(loc="best")
    add_title_subtitle("Variability vs length bucket",
                       "If oversubscription/paging occurs, p95/p99 will separate from the median dramatically")
    savefig(path)
    return path


def plot_compute_vs_e2e(df: pd.DataFrame, lat_col: str, gpu_col: str,
                        outdir: Path, logy: bool) -> Optional[Path]:
    if gpu_col is None or gpu_col not in df.columns:
        return None
    path = outdir / "05_compute_vs_e2e.png"
    plt.figure(figsize=(8, 6))
    plt.scatter(df[gpu_col], df[lat_col], s=10, alpha=0.5)
    plt.xlabel(f"{gpu_col} (ms)")
    plt.ylabel(f"{lat_col} (ms)")
    maybe_log_scale("y", logy)
    add_title_subtitle("GPU compute vs end-to-end latency",
                       "If points drift far above the diagonal trend, overhead (transfers/paging) dominates")
    savefig(path)
    return path


def plot_breakdown_by_bucket(df: pd.DataFrame, tok_col: str, lat_col: str,
                             bucket_col: str, h2d_col: Optional[str], d2h_col: Optional[str],
                             gpu_col: Optional[str], enq_col: Optional[str],
                             outdir: Path) -> Optional[Path]:
    have_any = any([c for c in [h2d_col, d2h_col, gpu_col, enq_col] if c in df.columns])
    if not have_any:
        return None

    path = outdir / "06_breakdown_stacked_by_bucket.png"

    # summarize medians by bucket
    cols = [c for c in [enq_col, h2d_col, gpu_col, d2h_col] if c and c in df.columns]
    agg = df.groupby(bucket_col)[cols + [lat_col]].median().reset_index()
    order = df.groupby(bucket_col)[tok_col].median().sort_values().index
    agg = agg.set_index(bucket_col).loc[order].reset_index()

    x = np.arange(len(agg))
    bottom = np.zeros(len(agg))
    plt.figure(figsize=(12, 6))

    for c in cols:
        plt.bar(x, agg[c].values, bottom=bottom, label=f"median {c}")
        bottom += agg[c].values

    # overlay median end-to-end line
    plt.plot(x, agg[lat_col].values, marker="o", linewidth=2, label=f"median {lat_col}")

    plt.xticks(x, [str(b) for b in agg[bucket_col]], rotation=30, ha="right")
    plt.xlabel("Length bucket")
    plt.ylabel("ms")
    plt.legend(fontsize=8, loc="best")
    add_title_subtitle("Median latency breakdown by bucket",
                       "Stacked bars = components; line = total. Gaps suggest unmeasured overhead (e.g., paging/allocator)")
    savefig(path)
    return path


def plot_throughput_by_bucket(df: pd.DataFrame, tok_col: str, lat_col: str,
                              bucket_col: str, outdir: Path) -> Path:
    path = outdir / "07_throughput_tokens_per_s_by_bucket.png"

    # tokens per second for a single-round (1 token decode) can be misleading;
    # still useful as an inverse latency view.
    df = df.copy()
    df["tok_per_s"] = 1000.0 / pd.to_numeric(df[lat_col], errors="coerce")

    agg = df.groupby(bucket_col).agg(
        median_tok_s=("tok_per_s", "median"),
        p10_tok_s=("tok_per_s", lambda x: np.nanpercentile(x, 10)),
        p90_tok_s=("tok_per_s", lambda x: np.nanpercentile(x, 90)),
        n=("tok_per_s", "count"),
    ).reset_index()

    order = df.groupby(bucket_col)[tok_col].median().sort_values().index
    agg = agg.set_index(bucket_col).loc[order].reset_index()

    x = np.arange(len(agg))
    plt.figure(figsize=(11, 6))
    plt.plot(x, agg["median_tok_s"], marker="o", label="median")
    plt.plot(x, agg["p10_tok_s"], marker="o", label="p10 (worst-ish)")
    plt.plot(x, agg["p90_tok_s"], marker="o", label="p90 (best-ish)")
    plt.xticks(x, [str(b) for b in agg[bucket_col]], rotation=30, ha="right")
    plt.xlabel("Length bucket")
    plt.ylabel("tokens/sec (1 / latency)")
    plt.legend(loc="best")
    add_title_subtitle("Throughput view by bucket (inverse latency)",
                       "Large drops at higher context length may indicate memory pressure effects")
    savefig(path)
    return path


def plot_cv_vs_tokens(df: pd.DataFrame, tok_col: str, lat_col: str,
                      bucket_col: str, outdir: Path) -> Path:
    path = outdir / "08_latency_cv_vs_tokens.png"
    agg = df.groupby(bucket_col).agg(
        mean_ms=(lat_col, "mean"),
        std_ms=(lat_col, "std"),
        med_tokens=(tok_col, "median"),
        n=(lat_col, "count"),
    ).reset_index()
    agg["cv"] = agg["std_ms"] / agg["mean_ms"]
    agg = agg.sort_values("med_tokens")

    plt.figure(figsize=(9, 6))
    plt.plot(agg["med_tokens"], agg["cv"], marker="o")
    plt.xlabel(f"median {tok_col} in bucket")
    plt.ylabel("CV = std/mean of latency")
    add_title_subtitle("Latency unpredictability (CV) vs length",
                       "Paging/thrashing often shows up as rising variability, not just rising median")
    savefig(path)
    return path


def write_outliers_and_summary(df: pd.DataFrame, tok_col: str, lat_col: str,
                               bucket_col: str, outdir: Path) -> Tuple[Path, Path]:
    outliers_path = outdir / "09_latency_outliers_table.csv"
    summary_path = outdir / "summary_stats_by_bucket.csv"

    df = df.copy()
    df["is_outlier"] = robust_outlier_mask(df[lat_col])

    # write outliers
    outliers = df[df["is_outlier"]].sort_values(lat_col, ascending=False)
    outliers.to_csv(outliers_path, index=False)

    # summary
    g = df.groupby(bucket_col)[lat_col]
    summary = pd.DataFrame({
        "n": g.count(),
        "mean_ms": g.mean(),
        "std_ms": g.std(),
        "median_ms": g.median(),
        "p90_ms": g.quantile(0.90),
        "p95_ms": g.quantile(0.95),
        "p99_ms": g.quantile(0.99),
        "min_ms": g.min(),
        "max_ms": g.max(),
    }).reset_index()

    # add token stats for ordering
    tok_stats = df.groupby(bucket_col)[tok_col].agg(["min", "median", "max"]).reset_index()
    tok_stats.columns = [bucket_col, "min_tokens", "median_tokens", "max_tokens"]
    summary = summary.merge(tok_stats, on=bucket_col, how="left").sort_values("median_tokens")

    summary.to_csv(summary_path, index=False)
    return outliers_path, summary_path


def write_html_index(outdir: Path, images: List[Path], extra_files: List[Path]) -> Path:
    idx_path = outdir / "index.html"
    rows = []
    for img in images:
        if img is None:
            continue
        rel = img.name
        rows.append(f"<h3>{rel}</h3><img src='{rel}' style='max-width: 100%; border:1px solid #ddd;'/>")

    extras = "".join([f"<li><a href='{f.name}'>{f.name}</a></li>" for f in extra_files if f is not None])
    html = f"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>LLM Measurements Plots</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 18px; }}
    code {{ background: #f2f2f2; padding: 2px 6px; border-radius: 6px; }}
  </style>
</head>
<body>
  <h1>LLM Measurements Plot Suite</h1>
  <p>Generated from <code>measurements.csv</code>. Use these figures in your report to discuss
     latency scaling, variability, and potential memory pressure effects.</p>
  <h2>Files</h2>
  <ul>{extras}</ul>
  <h2>Figures</h2>
  {''.join(rows)}
</body>
</html>
"""
    idx_path.write_text(html)
    return idx_path


# ---------------------------- main ----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to measurements.csv")
    ap.add_argument("--outdir", default="plots", help="Output directory for plots")
    ap.add_argument("--tok-col", default=None, help="Override token/context length column name")
    ap.add_argument("--lat-col", default=None, help="Override latency column name (end-to-end)")
    ap.add_argument("--bucket-col", default=None, help="Override bucket column; if missing, auto-create")
    ap.add_argument("--group-cols", default="", help="Comma-separated columns to color/group by (optional)")
    ap.add_argument("--logy", action="store_true", help="Use log scale on latency axis where relevant")
    ap.add_argument("--max-buckets", type=int, default=10, help="Max buckets to use when auto-bucketing")
    args = ap.parse_args()

    csv_path = Path(args.csv)
    outdir = Path(args.outdir)
    ensure_outdir(outdir)

    df = pd.read_csv(csv_path)
    if df.empty:
        raise SystemExit("CSV is empty.")

    # auto-detect key columns
    tok_col = args.tok_col or pick_first_existing(df, CAND_TOK_COLS)
    lat_col = args.lat_col or pick_first_existing(df, CAND_LAT_COLS)
    if tok_col is None:
        raise SystemExit(f"Could not find token/context length column. Looked for: {CAND_TOK_COLS}")
    if lat_col is None:
        raise SystemExit(f"Could not find latency column. Looked for: {CAND_LAT_COLS}")

    # numeric clean
    df[tok_col] = pd.to_numeric(df[tok_col], errors="coerce")
    df[lat_col] = pd.to_numeric(df[lat_col], errors="coerce")
    df = df.dropna(subset=[tok_col, lat_col]).copy()
    df = df[df[tok_col] > 0]

    # bucket column
    bucket_col = args.bucket_col or pick_first_existing(df, CAND_BUCKET_COLS)
    if bucket_col is None:
        # create buckets from token length (quantiles, up to max-buckets)
        nb = min(args.max_buckets, max(3, int(round(math.sqrt(len(df)) / 2))))
        df["bucket"] = quantile_buckets(df[tok_col], buckets=nb).astype(str)
        bucket_col = "bucket"

    # optional component columns
    gpu_col = pick_first_existing(df, CAND_GPU_COLS)
    h2d_col = pick_first_existing(df, CAND_H2D_COLS)
    d2h_col = pick_first_existing(df, CAND_D2H_COLS)
    enq_col = pick_first_existing(df, CAND_ENQ_COLS)

    group_cols = [c.strip() for c in args.group_cols.split(",") if c.strip()]

    # plots
    images: List[Optional[Path]] = []
    images.append(plot_scatter_latency_vs_tokens(df, tok_col, lat_col, outdir, args.logy, group_cols))
    images.append(plot_box_latency_by_bucket(df, tok_col, lat_col, bucket_col, outdir, args.logy))
    images.append(plot_hist_latency_by_bucket(df, tok_col, lat_col, bucket_col, outdir))
    images.append(plot_variability_by_bucket(df, tok_col, lat_col, bucket_col, outdir))
    images.append(plot_compute_vs_e2e(df, lat_col, gpu_col, outdir, args.logy))
    images.append(plot_breakdown_by_bucket(df, tok_col, lat_col, bucket_col, h2d_col, d2h_col, gpu_col, enq_col, outdir))
    images.append(plot_throughput_by_bucket(df, tok_col, lat_col, bucket_col, outdir))
    images.append(plot_cv_vs_tokens(df, tok_col, lat_col, bucket_col, outdir))

    outliers_path, summary_path = write_outliers_and_summary(df, tok_col, lat_col, bucket_col, outdir)

    idx = write_html_index(outdir, [p for p in images if p is not None], [outliers_path, summary_path])
    print(f"✅ Wrote plots to: {outdir}")
    print(f"✅ HTML index: {idx}")
    print(f"✅ Summary stats: {summary_path}")
    print(f"✅ Outliers table: {outliers_path}")
    print("\nDetected columns:")
    print(f"  token/context length: {tok_col}")
    print(f"  end-to-end latency:   {lat_col}")
    print(f"  bucket:               {bucket_col}")
    print(f"  gpu_col:              {gpu_col}")
    print(f"  enqueue_col:          {enq_col}")
    print(f"  h2d_col:              {h2d_col}")
    print(f"  d2h_col:              {d2h_col}")


if __name__ == "__main__":
    main()
