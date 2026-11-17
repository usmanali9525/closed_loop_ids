"""Figures and tables for the paper, from run_poisoning.py's poisoning_folds.csv.

Usage:
  python make_figures.py --folds results/poisoning_folds.csv --features raw --out figures_raw
  python make_figures.py --folds results_improved/poisoning_folds.csv --features improved \
      --out figures_improved

Outputs (in --out):
  fig1_f1_vs_error_rate.png      F1 vs share of wrong labels, one line per defence
  fig2_side_effects.png          FPR under random errors, recall under missed attacks
  fig3_labels_accepted.png       labels each defence actually learned from
  fig4_targeted.png              per-segment recall before/after targeted poisoning
  table_main.csv / .md           F1, FPR, recall (mean ± sd) for every scenario and defence
  table_targeted.csv / .md       paired targeted-poisoning analysis
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

ORDER = ["none", "frozen", "filter", "clip", "rollback", "memory", "trim"]
NAMES = {"none": "No defence", "frozen": "Frozen (no updates)", "filter": "Filter",
         "clip": "Clip", "rollback": "Rollback", "memory": "Memory refit",
         "trim": "Trimmed memory refit"}
STYLE = {"none": dict(color="#c0392b", marker="o"), "frozen": dict(color="#7f8c8d", ls="--"),
         "filter": dict(color="#2980b9", marker="s"), "clip": dict(color="#8e44ad", marker="^"),
         "rollback": dict(color="#27ae60", marker="D"), "memory": dict(color="#e67e22", marker="v"),
         "trim": dict(color="#16a085", marker="P", lw=2.2)}


def family(defence: str) -> str:
    return defence.split("@")[0].split("+")[0]


def label(defence: str) -> str:
    fam = family(defence)
    return NAMES.get(fam, defence) + (f" ({defence})" if "@" in defence else "")


def defence_order(defences):
    return sorted(defences, key=lambda d: (ORDER.index(family(d)) if family(d) in ORDER else 99, d))


def mean_ci(v):
    v = np.asarray(v, float)
    v = v[~np.isnan(v)]
    if len(v) < 2:
        return (v.mean() if len(v) else np.nan), np.nan
    return v.mean(), stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))


def curve(df, mode, metric, defence):
    """Mean and 95% CI of `metric` vs error rate; rate 0 is the clean scenario."""
    xs, ms, cs = [], [], []
    clean = df[(df.corruption == "none") & (df.defence == defence)]
    m, c = mean_ci(clean[metric]); xs.append(0.0); ms.append(m); cs.append(c)
    sub = df[(df.corruption == mode) & (df.defence == defence)]
    for r in sorted(sub.rate.unique()):
        m, c = mean_ci(sub[sub.rate == r][metric]); xs.append(r); ms.append(m); cs.append(c)
    return np.array(xs), np.array(ms), np.array(cs)


def line_panel(ax, df, mode, metric, defences, title, ylabel):
    for d in defences:
        x, m, c = curve(df, mode, metric, d)
        st = STYLE.get(family(d), {})
        ax.plot(100 * x, m, label=label(d), **st)
        ax.fill_between(100 * x, m - c, m + c, color=st.get("color", "k"), alpha=0.08)
    ax.set_title(title); ax.set_xlabel("Wrong analyst labels (%)"); ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)


def to_markdown(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(str(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", required=True, help="poisoning_folds.csv")
    ap.add_argument("--features", default="raw")
    ap.add_argument("--out", default="figures")
    ap.add_argument("--detectable", type=float, default=0.1,
                    help="targeted analysis keeps segments with clean recall >= this (no defence)")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(a.folds)
    df = df[df.features == a.features].copy()
    if df.empty:
        raise SystemExit(f"No rows for features='{a.features}' in {a.folds}")
    defences = defence_order(df.defence.unique())
    modes = [m for m in ("random", "missed_attack") if m in df.corruption.unique()]
    plt.rcParams.update({"font.size": 10, "figure.dpi": 150})

    # Figure 1: F1 vs error rate
    fig, axes = plt.subplots(1, len(modes), figsize=(5.2 * len(modes), 4), squeeze=False)
    for ax, mode in zip(axes[0], modes):
        line_panel(ax, df, mode, "F1", defences,
                   "Random label errors" if mode == "random" else "Missed attacks", "F1")
    axes[0][0].legend(fontsize=8, loc="lower left")
    fig.tight_layout(); fig.savefig(out / "fig1_f1_vs_error_rate.png"); plt.close(fig)

    # Figure 2: side effects (FPR for random errors, recall for missed attacks)
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4))
    if "random" in modes:
        line_panel(axes[0], df, "random", "FPR", defences, "False alarms (random errors)", "FPR")
    if "missed_attack" in modes:
        line_panel(axes[1], df, "missed_attack", "recall", defences,
                   "Detected attacks (missed-attack errors)", "Recall")
    axes[0].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out / "fig2_side_effects.png"); plt.close(fig)

    # Figure 3: labels actually used
    learners = [d for d in defences if family(d) != "frozen"]
    fig, axes = plt.subplots(1, len(modes), figsize=(5.2 * len(modes), 4), squeeze=False)
    for ax, mode in zip(axes[0], modes):
        line_panel(ax, df, mode, "n_labels_used", learners,
                   "Random label errors" if mode == "random" else "Missed attacks",
                   "Labels learned from (of 256)")
    axes[0][0].legend(fontsize=8, loc="lower left")
    fig.tight_layout(); fig.savefig(out / "fig3_labels_accepted.png"); plt.close(fig)

    # Main table
    rows = []
    mode_rank = {"none": 0, "random": 1, "missed_attack": 2}
    groups = sorted(df[df.corruption != "targeted"].groupby(["corruption", "rate"]),
                    key=lambda kv: (mode_rank.get(kv[0][0], 9), kv[0][1]))
    for (mode, rate), grp in groups:
        for d in defences:
            g = grp[grp.defence == d]
            if g.empty:
                continue
            rows.append({"scenario": "clean" if mode == "none" else f"{mode} {int(round(100 * rate))}%",
                         "defence": d,
                         **{m: f"{g[m].mean():.3f} ± {g[m].std(ddof=1):.3f}"
                            for m in ("F1", "FPR", "recall")},
                         "labels used": f"{g['n_labels_used'].mean():.0f}"})
    main = pd.DataFrame(rows)
    main.to_csv(out / "table_main.csv", index=False)
    (out / "table_main.md").write_text(to_markdown(main), encoding="utf-8")

    # Targeted: paired per (rep, fold, target segment), detectable segments only
    tg = df[df.corruption == "targeted"].dropna(subset=["target_id"])
    if not tg.empty:
        key = ["rep", "fold", "target_id"]
        wide = tg.pivot_table(index=key, columns=["defence", "rate"], values="target_recall")
        clean_none = wide[("none", 0.0)]
        keep = clean_none[clean_none >= a.detectable].index
        print(f"Targeted analysis: {len(keep)} of {len(clean_none)} segment-fold cases are "
              f"detectable when clean (recall >= {a.detectable})")
        trows, plot = [], []
        for d in defences:
            if (d, 0.0) not in wide or (d, 1.0) not in wide:
                continue
            before = wide.loc[keep, (d, 0.0)]; after = wide.loc[keep, (d, 1.0)]
            diff = (after - before).dropna()
            p = (float(stats.wilcoxon(diff).pvalue)
                 if len(diff) >= 6 and not np.allclose(diff, 0) else np.nan)
            nb = tg[(tg.defence == d) & (tg.rate == 1.0)]
            trows.append({"defence": d, "cases": len(diff),
                          "target recall clean": f"{before.mean():.3f}",
                          "target recall poisoned": f"{after.mean():.3f}",
                          "change": f"{diff.mean():+.3f}", "p (Wilcoxon)": f"{p:.4f}",
                          "other attacks recall (poisoned)": f"{nb['nontarget_recall'].mean():.3f}",
                          "overall F1 (poisoned)": f"{nb['F1'].mean():.3f}"})
            plot.append((d, before.mean(), after.mean()))
        tt = pd.DataFrame(trows)
        tt.to_csv(out / "table_targeted.csv", index=False)
        (out / "table_targeted.md").write_text(to_markdown(tt), encoding="utf-8")

        fig, ax = plt.subplots(figsize=(6.5, 4))
        x = np.arange(len(plot)); w = 0.38
        ax.bar(x - w / 2, [b for _, b, _ in plot], w, label="Clean labels", color="#95a5a6")
        ax.bar(x + w / 2, [p for _, _, p in plot], w, label="Target poisoned", color="#c0392b")
        ax.set_xticks(x, [NAMES.get(family(d), d) for d, _, _ in plot], rotation=15)
        ax.set_ylabel("Recall on the targeted attack"); ax.legend(); ax.grid(axis="y", alpha=0.3)
        ax.set_title("Targeted poisoning (attacks detectable when clean)")
        fig.tight_layout(); fig.savefig(out / "fig4_targeted.png"); plt.close(fig)

    print(f"Saved figures and tables to {out}/")
    print(main.to_string(index=False))
    if not tg.empty:
        print(); print(tt.to_string(index=False))


if __name__ == "__main__":
    main()
