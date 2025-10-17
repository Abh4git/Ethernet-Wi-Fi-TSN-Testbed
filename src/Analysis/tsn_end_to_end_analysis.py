#!/usr/bin/env python3
"""
tsn_end_to_end_analysis.py
For analysis of Latency over a pair of FIFO-TAS
For changing to your folders: Modify default Path on lines 360 and 361 (FIFO and TAS folders)
Given two folders of raw per-packet CSVs (FIFO and TAS), this script:
1) Matches files by name across folders.
2) For each matched pair and each "type" (by default the `payload` column), computes:
   - Latency stats: count, mean, median, std (sample) on `latency_ms`
   - Deadline misses: count & rate where latency_ms > --deadline_ms
   - Packet loss: expected_total − received_count  (see inference below)
   - Combined miss: deadline_miss_count + loss_count  (+ rates)
   - Deltas = FIFO − TAS (positive ⇒ TAS lower/better)
3) Across runs, performs per-type significance:
   - Wilcoxon signed-rank on paired median latencies
     * Hodges–Lehmann (HL) estimator of median difference (FIFO−TAS)
     * Bootstrap 95% CI for median difference
   - Two-proportion z-tests with Wilson 95% CIs for deadline, loss, combined

CSV schema (auto-detected delimiter; header required)
-----------------------------------------------------
arrival_time, src_ip, port, payload, latency_ms
(If 'payload' missing, use --group_by all to treat the whole file as one group.)

Expected total & loss inference
-------------------------------
- If you pass --expected_total N, we use N for both FIFO and TAS for each (file,type).
- Else if --infer_expected_total (default ON):
    expected_total = max(FIFO_count, TAS_count) for that (file,type).
- Else:
    expected_total = received_count (i.e., loss=0 by construction).

Outputs
-------
- per_run.csv          : per matched file & type with all stats and deltas
- latency_wilcoxon.csv : Wilcoxon + HL + bootstrap CI per type (and optional ALL)
- proportions_tests.csv: Wilson CIs + two-proportion z-tests per type (deadline/loss/combined)

Usage
-----
python tsn_end_to_end_analysis.py \
  --fifo_dir ./fifo_logs \
  --tas_dir  ./tas_logs  \
  --pattern  "*.csv" \
  --group_by payload \
  --deadline_ms 100.0 \
  --infer_expected_total \
  --include_all
"""

import argparse
import math
from pathlib import Path
from typing import Dict, Tuple, Optional, List

import numpy as np
import pandas as pd

# SciPy optional (for Wilcoxon p-values)
try:
    from scipy.stats import wilcoxon
    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False


# ---------------------- IO helpers ----------------------
def read_latency_df(csv_path: Path) -> pd.DataFrame:
    """Read CSV (auto-detect delimiter), ensure numeric latency, strip header whitespace."""
    try:
        df = pd.read_csv(csv_path, sep=None, engine='python')
    except Exception:
        try:
            df = pd.read_csv(csv_path)
        except Exception:
            df = pd.read_csv(csv_path, sep='\t')
    df.columns = [str(c).strip() for c in df.columns]
    if 'latency_ms' not in df.columns:
        raise ValueError(f"{csv_path} missing 'latency_ms'. Columns: {df.columns.tolist()}")
    df['latency_ms'] = pd.to_numeric(df['latency_ms'], errors='coerce')
    df = df.dropna(subset=['latency_ms'])
    return df


# ---------------------- Per-run stats ----------------------
def compute_latency_stats(df: pd.DataFrame) -> Dict[str, float]:
    s = df['latency_ms']
    n = int(s.count())
    return dict(
        count=n,
        mean=float(s.mean()) if n > 0 else float('nan'),
        median=float(s.median()) if n > 0 else float('nan'),
        std=float(s.std(ddof=1)) if n >= 2 else float('nan'),
    )

def count_deadline_misses(df: pd.DataFrame, deadline_ms: Optional[float]) -> int:
    if deadline_ms is None:
        return 0
    return int((df['latency_ms'] > deadline_ms).sum())


# ---------------------- Estimation & tests ----------------------
def hodges_lehmann_paired(diff: np.ndarray) -> float:
    """HL estimator: median of Walsh averages for paired differences."""
    diff = np.asarray(diff, dtype=float)
    diff = diff[np.isfinite(diff)]
    n = diff.size
    if n == 0:
        return float('nan')
    if n > 1000:  # speed shortcut
        return float(np.median(diff))
    walsh = []
    for i in range(n):
        for j in range(i, n):
            walsh.append(0.5 * (diff[i] + diff[j]))
    return float(np.median(walsh))

def bootstrap_ci_median(diff: np.ndarray, B: int = 10000, alpha: float = 0.05, seed: int = 12345) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    diff = np.asarray(diff, dtype=float)
    diff = diff[np.isfinite(diff)]
    n = diff.size
    if n == 0:
        return (float('nan'), float('nan'))
    idx = np.arange(n)
    meds = np.empty(B, dtype=float)
    for b in range(B):
        samp = diff[rng.choice(idx, size=n, replace=True)]
        meds[b] = np.median(samp)
    lo = np.percentile(meds, 2.5)
    hi = np.percentile(meds, 97.5)
    return float(lo), float(hi)

import math

def _safe_counts(k, n, label=""):
    """Clamp counts into [0, n] and coerce to ints; log-friendly hook via label."""
    try:
        k = int(round(k))
        n = int(n)
    except Exception:
        return 0, 0
    if n < 0: n = 0
    if k < 0: k = 0
    if k > n:
        # optional: print(f"[WARN] {label}: k ({k}) > n ({n}); clamping k to n.")
        k = n
    return k, n

def wilson_ci(k: int, n: int, alpha: float = 0.05):
    if n is None or k is None or n <= 0:
        return (float('nan'), float('nan'))
    k, n = _safe_counts(k, n)
    z = 1.959963984540054  # 95% two-sided
    p = k / n
    denom = 1.0 + (z*z)/n
    # guard tiny negatives from roundoff
    rad = (p*(1.0 - p))/n + (z*z)/(4.0*n*n)
    if rad < 0.0:
        rad = 0.0
    half   = (z * math.sqrt(rad)) / denom
    center = (p + (z*z)/(2.0*n)) / denom
    lo = max(0.0, center - half)
    hi = min(1.0, center + half)
    return (lo, hi)

def two_prop_z_test(x1: int, n1: int, x2: int, n2: int):
    """Two-sided test; pooled SE; robust to edge cases."""
    from math import erf, sqrt
    if min(n1, n2) is None or min(n1, n2) <= 0:
        return (float('nan'), float('nan'))
    x1, n1 = _safe_counts(x1, n1)
    x2, n2 = _safe_counts(x2, n2)

    p1 = x1 / n1
    p2 = x2 / n2
    p_pool = (x1 + x2) / (n1 + n2)  # guaranteed in [0,1] after _safe_counts

    se2 = p_pool*(1.0 - p_pool)*(1.0/n1 + 1.0/n2)
    # numeric guard (e.g., if p_pool is 0 or 1, se2==0)
    if se2 <= 0.0:
        if p1 == p2:
            return (0.0, 1.0)     # no difference, zero variance
        else:
            return (float('inf'), 0.0)  # different with zero variance → degenerate
    z = (p1 - p2) / math.sqrt(se2)
    # two-sided p-value via error function
    p = 2.0 * (1.0 - 0.5*(1.0 + erf(abs(z)/math.sqrt(2.0))))
    return (z, p)


# ---------------------- Build per-run table ----------------------
def build_per_run(fifo_dir: Path, tas_dir: Path, pattern: str, group_by: str,
                  deadline_ms: Optional[float], expected_total: Optional[int],
                  infer_expected_total: bool) -> pd.DataFrame:
    fifo_files = sorted(fifo_dir.glob(pattern))
    tas_files  = sorted(tas_dir.glob(pattern))
    tas_by_name = {p.name: p for p in tas_files}

    rows: List[Dict] = []
    matched = 0

    for fpath in fifo_files:
        name = fpath.name
        if name not in tas_by_name:
            continue
        tpath = tas_by_name[name]
        matched += 1

        df_f = read_latency_df(fpath)
        df_t = read_latency_df(tpath)

        # Ensure grouping column exists
        if group_by != 'all' and group_by not in df_f.columns:
            df_f[group_by] = 'all'
        if group_by != 'all' and group_by not in df_t.columns:
            df_t[group_by] = 'all'

        groups = ['all'] if group_by == 'all' else sorted(set(df_f[group_by].unique()) | set(df_t[group_by].unique()))

        for typ in groups:
            sub_f = df_f if group_by == 'all' else df_f[df_f[group_by] == typ]
            sub_t = df_t if group_by == 'all' else df_t[df_t[group_by] == typ]

            # Latency stats
            ls_f = compute_latency_stats(sub_f) if not sub_f.empty else dict(count=0, mean=np.nan, median=np.nan, std=np.nan)
            ls_t = compute_latency_stats(sub_t) if not sub_t.empty else dict(count=0, mean=np.nan, median=np.nan, std=np.nan)

            # Deadline misses
            dm_f = count_deadline_misses(sub_f, deadline_ms) if not sub_f.empty else 0
            dm_t = count_deadline_misses(sub_t, deadline_ms) if not sub_t.empty else 0

            # Expected totals for loss
            if expected_total is not None:
                E_f = E_t = int(expected_total)
            elif infer_expected_total:
                E_f = E_t = int(max(ls_f['count'], ls_t['count']))
            else:
                E_f = int(ls_f['count'])
                E_t = int(ls_t['count'])

            loss_f = max(0, E_f - ls_f['count'])
            loss_t = max(0, E_t - ls_t['count'])

            #comb_f = dm_f + loss_f
            #comb_t = dm_t + loss_t
            comb_f = min(dm_f + loss_f, E_f)
            comb_t = min(dm_t + loss_t, E_t)

            def safe_div(a, b): return (a / b) if b > 0 else np.nan

            rows.append(dict(
                file=name, type=typ,
                # latency
                fifo_count=ls_f['count'], fifo_mean=ls_f['mean'], fifo_median=ls_f['median'], fifo_std=ls_f['std'],
                tas_count=ls_t['count'],  tas_mean=ls_t['mean'],  tas_median=ls_t['median'],  tas_std=ls_t['std'],
                mean_delta=(ls_f['mean'] - ls_t['mean']) if np.isfinite(ls_f['mean']) and np.isfinite(ls_t['mean']) else np.nan,
                median_delta=(ls_f['median'] - ls_t['median']) if np.isfinite(ls_f['median']) and np.isfinite(ls_t['median']) else np.nan,
                std_delta=(ls_f['std'] - ls_t['std']) if np.isfinite(ls_f['std']) and np.isfinite(ls_t['std']) else np.nan,
                # reliability
                deadline_ms=deadline_ms,
                fifo_expected_total=E_f, tas_expected_total=E_t,
                fifo_deadline_miss_count=dm_f, tas_deadline_miss_count=dm_t,
                fifo_deadline_miss_rate=safe_div(dm_f, E_f), tas_deadline_miss_rate=safe_div(dm_t, E_t),
                fifo_loss_count=loss_f, tas_loss_count=loss_t,
                fifo_loss_rate=safe_div(loss_f, E_f), tas_loss_rate=safe_div(loss_t, E_t),
                fifo_combined_miss_count=comb_f, tas_combined_miss_count=comb_t,
                fifo_combined_miss_rate=safe_div(comb_f, E_f), tas_combined_miss_rate=safe_div(comb_t, E_t),
                # deltas (FIFO − TAS)
                dmr_delta=safe_div(dm_f, E_f) - safe_div(dm_t, E_t) if (E_f>0 and E_t>0) else np.nan,
                plr_delta=safe_div(loss_f, E_f) - safe_div(loss_t, E_t) if (E_f>0 and E_t>0) else np.nan,
                combined_delta=safe_div(comb_f, E_f) - safe_div(comb_t, E_t) if (E_f>0 and E_t>0) else np.nan,
                # paths
                fifo_path=str(fpath), tas_path=str(tpath),
            ))

    if matched == 0:
        raise SystemExit("No matching filenames found across FIFO and TAS with the given pattern.")
    return pd.DataFrame(rows)


# ---------------------- Significance (per type) ----------------------
def analyze_latencies_per_type(per_run_df: pd.DataFrame, B_boot: int) -> pd.DataFrame:
    rows = []
    for typ, g in per_run_df.groupby('type', dropna=False):
        sub = g.dropna(subset=['fifo_median', 'tas_median'])
        f = pd.to_numeric(sub['fifo_median'], errors='coerce')
        t = pd.to_numeric(sub['tas_median'], errors='coerce')
        mask = f.notna() & t.notna()
        d = (f[mask] - t[mask]).to_numpy()
        n = d.size
        if n == 0:
            rows.append(dict(type=typ, n_pairs=0, hl_median_delta=np.nan,
                             boot_ci_low=np.nan, boot_ci_high=np.nan,
                             wilcoxon_stat=np.nan, p_value=np.nan, direction='insufficient data'))
            continue
        hl = hodges_lehmann_paired(d)
        lo, hi = bootstrap_ci_median(d, B=B_boot)
        if SCIPY_AVAILABLE:
            try:
                res = wilcoxon(d, zero_method='pratt', alternative='two-sided', correction=False, mode='auto')
                w_stat = float(res.statistic); p_val = float(res.pvalue)
            except Exception:
                w_stat = np.nan; p_val = np.nan
        else:
            w_stat = np.nan; p_val = np.nan
        direction = 'TAS lower' if hl > 0 else ('FIFO lower' if hl < 0 else 'no change')
        rows.append(dict(type=typ, n_pairs=int(n), hl_median_delta=float(hl),
                         boot_ci_low=float(lo), boot_ci_high=float(hi),
                         wilcoxon_stat=w_stat, p_value=p_val, direction=direction))
    return pd.DataFrame(rows)

def sum_counts(df: pd.DataFrame, count_col: str, total_col: str) -> Tuple[int, int]:
    x = pd.to_numeric(df[count_col], errors='coerce').fillna(0).astype(int).sum()
    n = pd.to_numeric(df[total_col], errors='coerce').fillna(0).astype(int).sum()
    return int(x), int(n)

def analyze_proportions_per_type(per_run_df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for typ, g in per_run_df.groupby('type', dropna=False):
        for metric in ['deadline', 'loss', 'combined']:
            if metric == 'deadline':
                f_count, f_total = sum_counts(g, 'fifo_deadline_miss_count', 'fifo_expected_total')
                t_count, t_total = sum_counts(g, 'tas_deadline_miss_count',  'tas_expected_total')
            elif metric == 'loss':
                f_count, f_total = sum_counts(g, 'fifo_loss_count', 'fifo_expected_total')
                t_count, t_total = sum_counts(g, 'tas_loss_count',  'tas_expected_total')
            else:
                f_count, f_total = sum_counts(g, 'fifo_combined_miss_count', 'fifo_expected_total')
                t_count, t_total = sum_counts(g, 'tas_combined_miss_count',  'tas_expected_total')

            # BEFORE computing CIs/tests for each metric/type
            f_count, f_total = _safe_counts(f_count, f_total, label=f"{typ}/FIFO/{metric}")
            t_count, t_total = _safe_counts(t_count, t_total, label=f"{typ}/TAS/{metric}")

            f_lo, f_hi = wilson_ci(f_count, f_total)
            t_lo, t_hi = wilson_ci(t_count, t_total)
            z, p = two_prop_z_test(f_count, f_total, t_count, t_total)

            f_rate = f_count / f_total if f_total>0 else np.nan
            t_rate = t_count / t_total if t_total>0 else np.nan
            f_lo, f_hi = wilson_ci(f_count, f_total) if f_total>0 else (np.nan, np.nan)
            t_lo, t_hi = wilson_ci(t_count, t_total) if t_total>0 else (np.nan, np.nan)
            z, p = two_prop_z_test(f_count, f_total, t_count, t_total)

            out.append(dict(
                type=typ, metric=metric,
                fifo_count=int(f_count), fifo_total=int(f_total),
                fifo_rate=float(f_rate), fifo_wilson_lo=float(f_lo), fifo_wilson_hi=float(f_hi),
                tas_count=int(t_count), tas_total=int(t_total),
                tas_rate=float(t_rate), tas_wilson_lo=float(t_lo), tas_wilson_hi=float(t_hi),
                diff=float(f_rate - t_rate) if np.isfinite(f_rate) and np.isfinite(t_rate) else np.nan,
                z=float(z), p_value=float(p)
            ))
    return pd.DataFrame(out)


# ---------------------- CLI ----------------------
def main():
    ap = argparse.ArgumentParser(description="TSN end-to-end: per-run stats + Wilcoxon + two-proportion z with Wilson CIs")
    ap.add_argument('--fifo_dir', type=Path, default=Path("C:/results/5.1m/50Mbps/fifo/los/set1"))
    ap.add_argument('--tas_dir',  type=Path, default=Path("C:/results/5.1m/50Mbps/tas/los/set1"))
    ap.add_argument('--pattern',  type=str, default='*.csv')
    ap.add_argument('--group_by', type=str, choices=['payload','port','all'], default='payload')
    ap.add_argument('--deadline_ms', type=float, default=10.0, help='Threshold for deadline miss (ms)')
    ap.add_argument('--expected_total', type=int, default=5000, help='Global expected packets per (file,type)')
    ap.add_argument('--infer_expected_total', action='store_true', default=True,
                    help='Expected = max(FIFO_count, TAS_count) per (file,type) (default ON)')
    ap.add_argument('--no_infer_expected_total', action='store_false', dest='infer_expected_total',
                    help='Disable inference; expected=received unless --expected_total is set')
    ap.add_argument('--bootstrap', type=int, default=10000, help='Bootstrap iterations for median CI')
    ap.add_argument('--include_all', action='store_true', help='Add an ALL row aggregating over types')
    ap.add_argument('--out_per_run', default='per_run.csv')
    ap.add_argument('--out_latency', default='latency_wilcoxon.csv')
    ap.add_argument('--out_props',   default='proportions_tests.csv')
    args = ap.parse_args()

    per_run = build_per_run(args.fifo_dir, args.tas_dir, args.pattern, args.group_by,
                            args.deadline_ms, args.expected_total, args.infer_expected_total)
    per_run.to_csv(args.out_per_run, index=False)
    print(f"[OK] per-run -> {args.out_per_run} (rows: {len(per_run)})")

    lat = analyze_latencies_per_type(per_run, B_boot=args.bootstrap)
    if args.include_all:
        all_df = per_run.copy(); all_df['type'] = 'ALL'
        lat_all = analyze_latencies_per_type(all_df, B_boot=args.bootstrap)
        lat = pd.concat([lat, lat_all[lat_all['type']=='ALL']], ignore_index=True)
    lat.to_csv(args.out_latency, index=False)
    print(f"[OK] latency tests -> {args.out_latency}")

    props = analyze_proportions_per_type(per_run)
    if args.include_all:
        all_df = per_run.copy(); all_df['type'] = 'ALL'
        props_all = analyze_proportions_per_type(all_df)
        props = pd.concat([props, props_all[props_all['type']=='ALL']], ignore_index=True)
    props.to_csv(args.out_props, index=False)
    print(f"[OK] proportion tests -> {args.out_props}")

    # Console peek
    print("\n=== per-run (head) ===")
    print(per_run.head(10).to_string(index=False))
    print("\n=== latency tests ===")
    print(lat.to_string(index=False))
    print("\n=== proportion tests (head) ===")
    print(props.head(9).to_string(index=False))
    if not SCIPY_AVAILABLE:
        print("\n[NOTE] SciPy not available; Wilcoxon p-values/stat are NaN. Install scipy to enable them.")

if __name__ == '__main__':
    main()
