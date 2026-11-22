"""Report figures (aggregate only, except the synthetic patient mosaic)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ISO, LONG = "#9AA5B1", "#1F6FB2"


def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def fig_cohort(con, path):
    d = con.execute("SELECT subject_id, COUNT(*) n FROM admissions GROUP BY 1").df()
    fig, ax = plt.subplots(figsize=(6, 3.5))
    counts = d["n"].value_counts().sort_index()
    ax.bar(counts.index.astype(str), counts.values, color=LONG)
    ax.set(title="Admissions per patient (MIMIC-IV demo)", xlabel="Admissions", ylabel="Patients")
    ax.spines[["top", "right"]].set_visible(False)
    _save(fig, path)


def fig_h1(summary: pd.DataFrame, path):
    s = summary.reset_index()
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5), sharey=False)
    for ax, metric in zip(axes, ["auprc", "auroc"]):
        for i, model in enumerate(["logreg", "lightgbm"]):
            for j, (fs, col) in enumerate([("isolated", ISO), ("longitudinal", LONG)]):
                r = s[(s.model == model) & (s.features == fs)].iloc[0]
                ax.bar(i + (j - 0.5) * 0.38, r[f"{metric}_mean"], 0.36, yerr=r[f"{metric}_std"],
                       color=col, capsize=3, label=fs if i == 0 else None)
        ax.set_xticks([0, 1], ["Logistic regression", "LightGBM"])
        ax.set_title(metric.upper())
        ax.set_ylim(0, 1)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, loc="upper left")
    fig.suptitle("H1: isolated vs longitudinal (5x5 grouped CV, mean ± SD)")
    _save(fig, path)


def fig_h2(sub: pd.DataFrame, path, model="logreg"):
    d = sub[sub.model == model].pivot(index="group", columns="features", values="auprc")
    n = sub[(sub.model == model) & (sub.features == "isolated")].set_index("group")[["n", "pos"]]
    fig, ax = plt.subplots(figsize=(6, 3.5))
    x = range(len(d))
    ax.bar([i - 0.19 for i in x], d["isolated"], 0.36, color=ISO, label="isolated")
    ax.bar([i + 0.19 for i in x], d["longitudinal"], 0.36, color=LONG, label="longitudinal")
    ax.set_xticks(list(x), [f"{g}\n(n={int(n.loc[g,'n'])}, events={int(n.loc[g,'pos'])})" for g in d.index])
    ax.set(title=f"H2: AUPRC by prior admissions ({model})", ylabel="AUPRC", ylim=(0, 1))
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    _save(fig, path)


def fig_h3(imp: pd.DataFrame, path):
    d = imp.iloc[::-1]
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.barh(d["feature"], d["mean_abs_shap"],
            color=[LONG if t.startswith("history") else ISO for t in d["type"]])
    ax.set(title="H3: top features, longitudinal LightGBM (mean |SHAP|)", xlabel="mean |SHAP value|")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=LONG, label="history (cross-visit)"),
                       Patch(color=ISO, label="index admission")], frameon=False, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    _save(fig, path)


def fig_mosaic(tl: pd.DataFrame, path, title="Patient mosaic (synthetic patient)"):
    """One patient's fragments across admissions on a single time axis."""
    tl = tl.copy()
    tl["day"] = (pd.to_datetime(tl["event_time"]) - pd.to_datetime(tl["event_time"]).min()).dt.days
    rows = ["admission", "diagnosis", "lab", "medication", "icu_admit", "discharge"]
    colors = dict(zip(rows, ["#1F6FB2", "#7B61FF", "#2BA84A", "#F2A541", "#D64545", "#9AA5B1"]))
    fig, ax = plt.subplots(figsize=(9, 3.2))
    for i, et in enumerate(rows):
        d = tl[tl.event_type == et]
        ax.scatter(d["day"], [i] * len(d), s=14 if et == "lab" else 40, alpha=0.6, color=colors[et])
    ax.set_yticks(range(len(rows)), rows)
    ax.set(title=title, xlabel="Days since first event")
    ax.spines[["top", "right"]].set_visible(False)
    _save(fig, path)


# ---------------------------------------------------------------- reliability
def fig_reliability(curves: dict, path, title="R1: calibration (out-of-fold, logistic regression)"):
    """curves: {label: reliability_curve DataFrame}. Diagonal = perfect calibration."""
    styles = {"isolated": (ISO, "o"), "longitudinal": (LONG, "o")}
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharey=True)
    for ax, stage in zip(axes, ["raw", "recalibrated"]):
        ax.plot([0, 1], [0, 1], ls="--", color="#BBBBBB", lw=1, label="perfect")
        top = 0.0
        for key, c in curves.items():
            fs, st = key
            if st != stage:
                continue
            col, mk = styles[fs]
            ax.errorbar(c["p_mean"], c["obs"], yerr=[c["obs"] - c["obs_lo"], c["obs_hi"] - c["obs"]],
                        color=col, marker=mk, ms=4, lw=1.2, capsize=2, label=fs)
            top = max(top, c["p_mean"].max(), c["obs_hi"].max())
        lim = min(1.0, top * 1.1 + 0.02)
        ax.set(xlim=(0, lim), ylim=(0, lim), xlabel="Predicted risk",
               title="As trained (class_weight='balanced')" if stage == "raw" else "After cross-fitted Platt")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Observed rate (95% CI)")
    axes[0].legend(frameon=False, loc="upper left")
    fig.suptitle(title)
    _save(fig, path)


def fig_bootstrap(deltas: dict, path, metric="AUPRC"):
    """deltas: {model: array of bootstrap deltas (longitudinal - isolated)}."""
    import numpy as np
    fig, axes = plt.subplots(1, len(deltas), figsize=(4.2 * len(deltas), 3.4), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, (model, d) in zip(axes, deltas.items()):
        lo, hi = np.percentile(d, [2.5, 97.5])
        ax.hist(d, bins=40, color=LONG, alpha=0.8)
        ax.axvline(0, color="#333333", lw=1)
        ax.axvspan(lo, hi, color=LONG, alpha=0.12)
        ax.set(title=f"{model}: Δ{metric} 95% CI [{lo:+.3f}, {hi:+.3f}]",
               xlabel=f"Δ{metric} (longitudinal − isolated)")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Bootstrap resamples")
    fig.suptitle("R2: H1 gain under patient-level bootstrap")
    _save(fig, path)


def fig_conformal(summ, path, model="logreg"):
    """Stacked bars: share of admissions the model can call low / high / must call uncertain."""
    d = summ[summ.model == model]
    parts = [("share_low", "low risk {0}", "#2BA84A"), ("share_uncertain", "uncertain {0,1}", "#D5D9DE"),
             ("share_high", "high risk {1}", "#D64545")]
    labels = [f"{r.features}\nα={r.alpha:g}" for r in d.itertuples()]
    fig, ax = plt.subplots(figsize=(7, 3.6))
    left = [0.0] * len(d)
    for col, lab, colr in parts:
        vals = d[col].tolist()
        ax.barh(labels, vals, left=left, color=colr, label=lab)
        left = [a + b for a, b in zip(left, vals)]
    for i, r in enumerate(d.itertuples()):
        ax.text(1.01, i, f"coverage {r.coverage:.2f}", va="center", fontsize=8)
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set(xlim=(0, 1.25), xlabel="Share of index admissions", title=f"R3: conformal prediction sets ({model})")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.42, -0.2), ncol=3, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    _save(fig, path)
