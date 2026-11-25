#!/usr/bin/env python3
"""Analyse the benchmark results in data/*.csv: plots go to figures/, number tables to data/summary.md."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
FIG = ROOT / "figures"
FIG.mkdir(parents=True, exist_ok=True)

# ---- palette
SURFACE = "#fcfcfb"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
ALGO_COLOR = {"ijk": CAT[0], "ikj": CAT[1], "blocked": CAT[2]}
BLUE_RAMP = ["#86b6ef", "#5598e7", "#3987e5", "#2a78d6", "#1c5cab", "#104281", "#0d366b"]

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
    "lines.linewidth": 1.6, "lines.markersize": 5, "legend.frameon": False, "figure.dpi": 110,
})


def load(name):
    p = DATA / f"{name}.csv"
    return pd.read_csv(p) if p.exists() else None


def save(fig, name, title=None):
    if title:
        fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG / f"{name}.png", dpi=200)
    plt.close(fig)
    print("  figure:", name)


def ramp(values):
    idx = np.linspace(0, len(BLUE_RAMP) - 1, len(values)).round().astype(int)
    return {v: BLUE_RAMP[i] for v, i in zip(sorted(values), idx)}


# ------------------------------------------------------------------ Amdahl helpers

def amdahl_fit(p, s):
    p, s = np.asarray(p, float), np.asarray(s, float)
    mask = p > 1
    x, y = 1 - 1 / p[mask], 1 / s[mask] - 1 / p[mask]
    f = float((x * y).sum() / (x * x).sum())
    pred = 1 / (f + (1 - f) / p[mask])
    ss_res = ((s[mask] - pred) ** 2).sum()
    ss_tot = ((s[mask] - s[mask].mean()) ** 2).sum()
    return f, 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def amdahl(p, f):
    p = np.asarray(p, float)
    return 1 / (f + (1 - f) / p)


def karp_flatt(p, s):
    p, s = np.asarray(p, float), np.asarray(s, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(p > 1, (1 / s - 1 / p) / (1 - 1 / p), np.nan)


def add_speedup(df):
    out = []
    for (algo, n), g in df.groupby(["algo", "n"]):
        g = g.sort_values("threads").copy()
        if 1 not in set(g.threads):
            continue
        t1 = g.loc[g.threads == 1, "t_total"].iloc[0]
        m1 = g.loc[g.threads == 1, "t_mult_med"].iloc[0]
        g["S"] = t1 / g.t_total
        g["S_mult"] = m1 / g.t_mult_med
        g["E"] = g.S / g.threads
        g["KF"] = karp_flatt(g.threads, g.S)
        out.append(g)
    return pd.concat(out) if out else df.iloc[0:0]


# ------------------------------------------------------------------ scaling (2a, 2b, 4)

def scaling(lines):
    df = load("scaling")
    if df is None:
        return
    df = add_speedup(df)
    algos = [a for a in ["ijk", "ikj", "blocked"] if a in set(df.algo)]
    threads = sorted(df.threads.unique())

    # (1) multiply time vs threads
    fig, axes = plt.subplots(1, len(algos), figsize=(4.2 * len(algos), 3.6), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, algo in zip(axes, algos):
        g = df[df.algo == algo]
        colors = ramp(g.n.unique())
        for n, gn in g.groupby("n"):
            ax.plot(gn.threads, gn.t_mult_med, "o-", color=colors[n], label=f"n={n}")
        ax.set(xscale="log", yscale="log", title=algo, xlabel="threads (p)")
        ax.set_xticks(threads, [str(t) for t in threads])
        ax.minorticks_off()
    axes[0].set_ylabel("multiply time (s, median)")
    axes[-1].legend(fontsize=8, loc="upper right")
    save(fig, "01_time_vs_threads", "Multiply time vs number of threads (median, log–log)")

    # (2) speedup, the ideal line and the Amdahl fit
    fig, axes = plt.subplots(1, len(algos), figsize=(4.2 * len(algos), 3.8), sharey=True)
    axes = np.atleast_1d(axes)
    pp = np.linspace(1, max(threads), 200)
    for ax, algo in zip(axes, algos):
        g = df[df.algo == algo]
        colors = ramp(g.n.unique())
        ax.plot(pp, pp, ":", color=INK2, lw=1, label="ideal S=p")
        ax.axvline(16, color=GRID, lw=1.5, zorder=0)
        ax.text(16.4, 0.97, "16 physical cores →\nSMT beyond", fontsize=7, color=INK2,
                va="top", transform=ax.get_xaxis_transform())
        for n, gn in g.groupby("n"):
            ax.plot(gn.threads, gn.S, "o-", color=colors[n], label=f"n={n}")
        counts = g.groupby("n").threads.nunique()
        nmax = counts[counts >= 4].index.max()
        gm = g[g.n == nmax]
        f, r2 = amdahl_fit(gm.threads, gm.S)
        ax.plot(pp, amdahl(pp, f), "--", color=CAT[1], lw=1.4,
                label=f"Amdahl fit n={nmax}\nf={f:.3f}")
        ax.set(title=algo, xlabel="threads (p)", xlim=(0, max(threads) + 1))
    axes[0].set_ylabel("speedup S(p) = T(1)/T(p)")
    axes[-1].legend(fontsize=7, loc="upper left")
    save(fig, "02_speedup", "Speedup S(p) vs number of threads, with ideal line and Amdahl fit")

    # (3) efficiency
    fig, axes = plt.subplots(1, len(algos), figsize=(4.2 * len(algos), 3.4), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, algo in zip(axes, algos):
        g = df[df.algo == algo]
        colors = ramp(g.n.unique())
        for n, gn in g.groupby("n"):
            ax.plot(gn.threads, gn.E, "o-", color=colors[n], label=f"n={n}")
        ax.set(title=algo, xlabel="threads (p)", xscale="log", ylim=(0, 1.15))
        ax.set_xticks(threads, [str(t) for t in threads])
        ax.minorticks_off()
    axes[0].set_ylabel("efficiency E(p) = S/p")
    axes[-1].legend(fontsize=8)
    save(fig, "03_efficiency", "Parallel efficiency E(p) vs number of threads")

    # (4) GFLOP/s vs n
    panels = [t for t in (1, 16) if t in threads]
    fig, axes = plt.subplots(1, len(panels), figsize=(4.6 * len(panels), 3.6), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, t in zip(axes, panels):
        for algo in algos:
            g = df[(df.algo == algo) & (df.threads == t)].sort_values("n")
            ax.plot(g.n, g.gflops, "o-", color=ALGO_COLOR[algo], label=algo)
        for cache_bytes, name in [(1 << 20, "L2/core 1MB"), (32 << 20, "L3/CCD 32MB"),
                                  (64 << 20, "L3 total 64MB")]:
            n_fit = (cache_bytes / 24) ** 0.5
            ax.axvline(n_fit, color=INK2, ls=":", lw=1, zorder=0)
            ax.text(n_fit * 1.03, 0.98, name, rotation=90, va="top", fontsize=7, color=INK2,
                    transform=ax.get_xaxis_transform())
        ax.set(title=f"{t} thread{'s' if t > 1 else ''}", xscale="log", xlabel="matrix size n")
        ns = sorted(df.n.unique())
        ax.set_xticks(ns, [str(n) for n in ns], rotation=45)
        ax.minorticks_off()
    axes[0].set_ylabel("GFLOP/s (higher is better)")
    axes[-1].legend(fontsize=8)
    save(fig, "04_gflops_vs_n", "Throughput (GFLOP/s) vs matrix size n")

    # (9) Karp–Flatt
    fig, axes = plt.subplots(1, len(algos), figsize=(4.2 * len(algos), 3.4), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, algo in zip(axes, algos):
        g = df[(df.algo == algo) & (df.threads > 1) & (df.n >= 800)]
        colors = ramp(df[df.algo == algo].n.unique())
        for n, gn in g.groupby("n"):
            ax.plot(gn.threads, gn.KF, "o-", color=colors[n], label=f"n={n}")
        ax.set(title=algo, xlabel="threads (p)")
    axes[0].set_ylabel("Karp–Flatt e(p)")
    axes[-1].legend(fontsize=8)
    save(fig, "09_karp_flatt", "Karp–Flatt serial fraction e(p) vs number of threads (n ≥ 800)")

    # (8) memory
    g = df[(df.threads == 1) & (df.algo == "blocked")].sort_values("n")
    if len(g):
        fig, ax = plt.subplots(figsize=(5, 3.6))
        ax.plot(g.n, g.maxrss_kb / 1024, "o-", color=CAT[0], label="measured peak RSS")
        ax.plot(g.n, 3 * g.n**2 * 8 / 2**20, "--", color=CAT[1], label="theory 3 × n² × 8 B")
        ax.set(xscale="log", yscale="log", xlabel="matrix size n", ylabel="memory (MiB)",
               title="Peak memory vs matrix size (1 thread, blocked)")
        ax.set_xticks(g.n, [str(n) for n in g.n], rotation=45)
        ax.minorticks_off()
        ax.legend()
        save(fig, "08_memory_vs_n")
        nn = 1600 if 1600 in set(df.n) else df.n.max()
        g2 = df[(df.n == nn) & (df.algo == "blocked")].sort_values("threads")
        fig, ax = plt.subplots(figsize=(5, 3.6))
        ax.plot(g2.threads, g2.vmpeak_kb / 1024, "o-", color=CAT[0], label="peak virtual (VmPeak)")
        ax.plot(g2.threads, g2.maxrss_kb / 1024, "o-", color=CAT[1], label="peak resident (RSS)")
        ax.set(xlabel="threads (p)", ylabel="MiB", ylim=(0, None),
               title=f"Peak memory vs threads (n={nn}, blocked)")
        ax.legend()
        save(fig, "08b_memory_vs_threads")

    # ---- Amdahl table
    lines += ["## Amdahl / serial fraction (exp: scaling)", "",
              "f_fit(all) = fit over every p; f_fit(≤16) = fit over p ≤ 16 only (no SMT); "
              "f_meas = (t_init+t_verify)/T(1); S_max = highest measured speedup (at p); "
              "S_∞ = 1/f_fit(≤16) = Amdahl ceiling", "",
              "| algo | n | T(1) s | f_fit(all) | R² | f_fit(≤16) | f_meas | S_max (p) | S_∞ | KF@16 | KF@32 |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
    for algo in algos:
        for n, g in df[df.algo == algo].groupby("n"):
            if g.threads.nunique() < 3:
                continue
            f_all, r2 = amdahl_fit(g.threads, g.S)
            g16 = g[g.threads <= 16]
            f16, _ = amdahl_fit(g16.threads, g16.S)
            r1 = g[g.threads == 1].iloc[0]
            f_meas = (r1.t_init + r1.t_verify) / r1.t_total
            best = g.loc[g.S.idxmax()]
            kf = {p: g.loc[g.threads == p, "KF"] for p in (16, 32)}
            kfs = [f"{v.iloc[0]:.4f}" if len(v) else "n/a" for v in kf.values()]
            lines.append(f"| {algo} | {n} | {r1.t_total:.4f} | {f_all:.4f} | {r2:.3f} | "
                         f"{f16:.4f} | {f_meas:.4f} | {best.S:.2f} ({int(best.threads)}) | "
                         f"{1 / f16 if f16 > 0 else float('inf'):.1f} | {kfs[0]} | {kfs[1]} |")
    lines += ["", "### temperature (Tctl, °C) range during scaling runs", "",
              f"before: {df.temp_before.min():.1f} to {df.temp_before.max():.1f}; "
              f"after: {df.temp_after.min():.1f} to {df.temp_after.max():.1f}", ""]


# ------------------------------------------------------------------ program settings (2c)

def block(lines):
    df = load("block")
    if df is None:
        return
    combos = sorted({(n, t) for n, t in zip(df.n, df.threads)})
    fig, ax = plt.subplots(figsize=(6, 3.8))
    for i, (n, t) in enumerate(combos):
        g = df[(df.n == n) & (df.threads == t)].sort_values("block")
        ax.plot(g.block, g.gflops, "o-", color=CAT[i], label=f"n={n}, {t} thr")
    ax.set(xscale="log", xlabel="block (tile) size b", ylabel="GFLOP/s",
           title="blocked: effect of tile size")
    bs = sorted(df.block.unique())
    ax.set_xticks(bs, [str(b) for b in bs])
    ax.minorticks_off()
    ax.legend(fontsize=8)
    save(fig, "05_block_size")
    lines += ["## block size (best per n, threads)", ""]
    for (n, t), g in df.groupby(["n", "threads"]):
        b = g.loc[g.gflops.idxmax()]
        lines.append(f"- n={n}, {t} thr: best b={int(b.block)} → {b.gflops:.1f} GF/s "
                     f"(worst b={int(g.loc[g.gflops.idxmin()].block)} → {g.gflops.min():.1f})")
    lines.append("")


def grouped_bars(df, label_col, group_cols, title, fname, lines, ylabel="GFLOP/s"):
    labels = list(dict.fromkeys(df[label_col]))
    groups = list(dict.fromkeys(df[group_cols].astype(str).agg(", ".join, axis=1)))
    df = df.assign(_grp=df[group_cols].astype(str).agg(", ".join, axis=1))
    width = 0.8 / len(groups)
    fig, ax = plt.subplots(figsize=(max(6, 0.9 * len(labels) * max(1, len(groups) / 3)), 3.9))
    for i, grp in enumerate(groups):
        vals = [df[(df[label_col] == l) & (df._grp == grp)].gflops.mean() for l in labels]
        x = np.arange(len(labels)) + (i - (len(groups) - 1) / 2) * width
        ax.bar(x, vals, width * 0.92, color=CAT[i % len(CAT)], label=grp)
    ax.set_xticks(np.arange(len(labels)), labels, rotation=30, ha="right")
    ax.set(ylabel=ylabel, title=title)
    ax.grid(axis="x", visible=False)
    ax.legend(fontsize=7, ncol=min(4, len(groups)))
    save(fig, fname)
    lines += [f"## {title}", "", df.pivot_table(index=label_col, columns="_grp", values="gflops",
                                                 sort=False).round(1).to_markdown(), ""]


def schedule(lines):
    df = load("schedule")
    if df is None:
        return
    for n in sorted(df.n.unique()):
        grouped_bars(df[df.n == n], "label", ["algo", "threads"],
                     f"OpenMP schedule, n={n}", f"06_schedule_n{n}", lines)


def compiler(lines):
    df = load("compiler")
    if df is None:
        return
    grouped_bars(df, "build", ["algo", "threads"], "gcc optimisation level, n=1600",
                 "07_compiler", lines)


# ------------------------------------------------------------------ OS settings (2d)

def os_settings(lines):
    df = load("os")
    if df is None:
        return
    df = df.assign(kind=df.label.str.split(":").str[0], setting=df.label.str.split(":").str[1])
    titles = {"bind": "thread affinity (OMP_PROC_BIND / OMP_PLACES), n=3200",
              "taskset": "CPU set via taskset (SMT vs cores, 1 vs 2 CCDs), n=3200",
              "contention": "competing process (another 16-thread job), n=3200",
              "power": "power profile, n=3200"}
    for kind, title in titles.items():
        g = df[df.kind == kind]
        if len(g):
            grouped_bars(g, "setting", ["algo", "threads"], title, f"10_os_{kind}", lines)
    g = df[df.kind == "stack"]
    if len(g):
        lines += ["## OMP_STACKSIZE (n=1600, 32 threads, blocked)", "",
                  "| OMP_STACKSIZE | VmPeak MiB | peak RSS MiB | GFLOP/s |", "|---|---|---|---|"]
        for _, r in g.iterrows():
            lines.append(f"| {r.setting} | {r.vmpeak_kb / 1024:.0f} | {r.maxrss_kb / 1024:.0f} | "
                         f"{r.gflops:.1f} |")
        lines.append("")


def main():
    lines = ["# Benchmark summary (generated by bench/analyze.py; do not edit by hand)", ""]
    for step in (scaling, block, schedule, compiler, os_settings):
        step(lines)
    (DATA / "summary.md").write_text("\n".join(lines) + "\n")
    print("  summary: data/summary.md")


if __name__ == "__main__":
    main()
