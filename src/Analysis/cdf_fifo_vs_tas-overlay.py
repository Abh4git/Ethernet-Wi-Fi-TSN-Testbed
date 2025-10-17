#!/usr/bin/env python3
# Testbed Project
# Plots Cumulative Distribution Function(CDF)
# One by one for each Run across FIFO and TAS 

# To run from your folders: modify the default Path specified on Lines 130 and 131

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Colors for consistency across your work
TYPE_COLORS = {
    "video": "red",  
    "vitals": "green",    
    "ehr":  "blue",     
}

def read_log(path: Path) -> pd.DataFrame:
    """
    Reads a log file with columns:
      arrival_time, src_ip, port, payload, latency_ms
    Delimiter is inferred (comma, tab, or spaces).
    """
    df = pd.read_csv(path, sep=None, engine="python")
    # Normalize column names just in case
    df.columns = [c.strip().lower() for c in df.columns]
    # Ensure required columns exist
    required = {"arrival_time", "src_ip", "port", "payload", "latency_ms"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {', '.join(sorted(missing))}")
    # Coerce latency to float; drop rows without valid latency
    df["latency_ms"] = pd.to_numeric(df["latency_ms"], errors="coerce")
    df = df.dropna(subset=["latency_ms"])
    # Normalize payload strings
    df["payload"] = df["payload"].astype(str).str.strip().str.lower()
    return df

def compute_cdf(series: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """
    Given a 1D Series of numbers, return (x_sorted, y_percent)
    where y is 0..100 CDF.
    """
    x = np.sort(series.values)
    if x.size == 0:
        return x, np.array([])
    y = np.arange(1, x.size + 1) / x.size * 100.0
    return x, y

def plot_cdf_per_type(ax, df: pd.DataFrame, title: str):
    """
    Plot CDF curves for each traffic type present in `df` on the given axes.
    """
    types_present = sorted(df["payload"].dropna().unique())
    drew_any = False
    for t in ["video", "vitals", "ehr"]:
        if t in types_present:
            x, y = compute_cdf(df.loc[df["payload"] == t, "latency_ms"])
            if x.size:
                ax.plot(x, y, label=f"{t} (n={x.size})", linewidth=2,
                        color=TYPE_COLORS.get(t, None))
                drew_any = True
    # In case there are other payload types, plot them too
    for t in types_present:
        if t not in {"video", "vitals", "ehr"}:
            x, y = compute_cdf(df.loc[df["payload"] == t, "latency_ms"])
            if x.size:
                ax.plot(x, y, label=f"{t} (n={x.size})", linewidth=2)

    ax.set_title(title)
    ax.set_xlabel("Latency (ms)")
    ax.set_ylabel("Packets ≤ latency (%)")
    ax.grid(True, linestyle="--", alpha=0.4)
    if drew_any or len(types_present) > 0:
        # Put legend outside to keep the plot area clear
        ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)

#new
def plot_cdf_overlay(ax, fifo_df: pd.DataFrame, tas_df: pd.DataFrame, title: str):
    """
    Plot TAS CDF per traffic type, with FIFO as dashed reference.
    """
    types_present = sorted(set(fifo_df["payload"].unique()) | set(tas_df["payload"].unique()))

    for t in ["video", "vitals", "ehr"]:
        if t in types_present:
            # FIFO reference (dashed)
            if not fifo_df.empty:
                x_f, y_f = compute_cdf(fifo_df.loc[fifo_df["payload"] == t, "latency_ms"])
                if (x_f.size == 5001):
                    size1=5000
                elif (x_f.size == 25001):
                    size1 = 25000
                else:
                    size1=x_f.size
                if x_f.size:
                    ax.plot(x_f, y_f, linestyle="--", linewidth=1.5,
                            color=TYPE_COLORS.get(t, None),
                            label=f"FIFO {t} (n={size1})")

            # TAS main (solid)
            if not tas_df.empty:
                x_t, y_t = compute_cdf(tas_df.loc[tas_df["payload"] == t, "latency_ms"])
                if (x_f.size == 5001):
                    size2 = 5000
                elif (x_f.size == 25001):
                    size2 = 25000
                else:
                    size2 = x_f.size

                if x_t.size:
                    ax.plot(x_t, y_t, linestyle="-", linewidth=2,
                            color=TYPE_COLORS.get(t, None),
                            label=f"TAS {t} (n={size2})")

    ax.set_title(title)
    ax.set_xlabel("Latency (ms)")
    ax.set_ylabel("Packets ≤ latency (%)")
    ax.set_ylim(0, 100)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)

def main():
    parser = argparse.ArgumentParser(
        description="CDF of latency vs. percentage of packets for FIFO and TAS logs."
    )
    parser.add_argument("fifo_log", nargs="?",     type=Path, default=Path(r"C:\results\5.1\50Mbps\fifo\los\set1\traffic_log2.csv"), help="Path to FIFO log file")
    parser.add_argument("tas_log", nargs="?",    type=Path, default=Path(r"C:\results\5.1m\50Mbps\tas\los\set1\traffic_log2.csv"), help="Path to TAS log file")

    parser.add_argument("-o", "--out", type=Path, default=Path("../cdf_fifo_vs_tas.png"), help="Output image filename (PNG)")
    args = parser.parse_args()
    print(args.fifo_log)
    fifo_df = read_log(args.fifo_log)
    tas_df  = read_log(args.tas_log)

    # Determine a shared x-limit so both panes use the same scale
    xmax = float(
        np.nanmax([
            fifo_df["latency_ms"].max() if not fifo_df.empty else 0,
            tas_df["latency_ms"].max() if not tas_df.empty else 0
        ])
    )
    # Add a small headroom
    xmax = 15 #xmax * 1.02 if xmax > 0 else 1.0

    fig, ax = plt.subplots(figsize=(7, 5))
    plot_cdf_overlay(ax, fifo_df, tas_df, "TAS vs. FIFO Reference (Latency CDF)")

    # Shared x-limit
    # xmax = float(
    #     np.nanmax([
    #         fifo_df["latency_ms"].max() if not fifo_df.empty else 0,
    #         tas_df["latency_ms"].max() if not tas_df.empty else 0
    #     ])
    # )
    ax.set_xlim(0, xmax * 1.02 if xmax > 0 else 1.0)

    fig.suptitle("Latency CDF by Traffic Type\n(Solid = TAS, Dashed = FIFO)", y=1.02, fontsize=13)
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
