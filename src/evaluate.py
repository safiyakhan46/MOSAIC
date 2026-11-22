"""Robust evaluation for small data + hypothesis tests H1-H3.

- Repeated patient-grouped K-fold CV (out-of-fold predictions for every row)
- H1: paired difference in AUPRC/AUROC, longitudinal minus isolated, per repeat
- H2: performance by number of prior admissions (0 / 1-2 / 3+)
- H3: SHAP mean |value| for the longitudinal LightGBM model
"""
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

from src.train import ID_COLS, models


def oof_predictions(df: pd.DataFrame, cfg: dict, n_splits=5, n_repeats=5) -> pd.DataFrame:
    """Out-of-fold probabilities: rows x (repeat, model)."""
    X, y, g = df.drop(columns=ID_COLS), df["label"].values, df["subject_id"].values
    out = []
    for r in range(n_repeats):
        cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True,
                                  random_state=cfg["random_state"] + r)
        preds = {m: np.zeros(len(df)) for m in ("logreg", "lightgbm")}
        for tr, te in cv.split(X, y, g):
            for name, model in models(X.iloc[tr], cfg["random_state"]).items():
                model.fit(X.iloc[tr], y[tr])
                preds[name][te] = model.predict_proba(X.iloc[te])[:, 1]
        for name, p in preds.items():
            out.append(pd.DataFrame({"hadm_id": df["hadm_id"].values, "label": y,
                                     "repeat": r, "model": name, "p": p}))
    return pd.concat(out, ignore_index=True)


def _score(d: pd.DataFrame) -> pd.Series:
    if d["label"].nunique() < 2:
        return pd.Series({"auprc": np.nan, "auroc": np.nan, "n": len(d), "pos": d["label"].sum()})
    return pd.Series({"auprc": average_precision_score(d["label"], d["p"]),
                      "auroc": roc_auc_score(d["label"], d["p"]),
                      "n": len(d), "pos": d["label"].sum()})


def h1_compare(oof_iso: pd.DataFrame, oof_long: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    s_iso = oof_iso.groupby(["model", "repeat"]).apply(_score, include_groups=False)
    s_long = oof_long.groupby(["model", "repeat"]).apply(_score, include_groups=False)
    summary = pd.concat({"isolated": s_iso, "longitudinal": s_long}, names=["features"])
    summary = (summary.groupby(["features", "model"])[["auprc", "auroc"]]
               .agg(["mean", "std"]).round(3))
    summary.columns = [f"{m}_{s}" for m, s in summary.columns]
    diff = (s_long[["auprc", "auroc"]] - s_iso[["auprc", "auroc"]])
    paired = diff.groupby("model").agg(["mean", "std", "min", "max"]).round(3)
    paired.columns = [f"delta_{m}_{s}" for m, s in paired.columns]
    paired["repeats_improved_auprc"] = (diff["auprc"] > 0).groupby("model").sum().astype(str) + "/" \
        + diff["auprc"].groupby("model").size().astype(str)
    return summary, paired


def h2_subgroups(oof_iso, oof_long, longi: pd.DataFrame) -> pd.DataFrame:
    bins = pd.cut(longi["n_prior_adm"], [-1, 0, 2, np.inf], labels=["0 prior", "1-2 prior", "3+ prior"])
    grp = pd.DataFrame({"hadm_id": longi["hadm_id"], "group": bins})
    rows = []
    for fs, oof in [("isolated", oof_iso), ("longitudinal", oof_long)]:
        d = oof.merge(grp, on="hadm_id")
        s = d.groupby(["model", "group", "repeat"], observed=True).apply(_score, include_groups=False)
        s = s.groupby(["model", "group"], observed=True).agg(
            auprc=("auprc", "mean"), auroc=("auroc", "mean"), n=("n", "first"), pos=("pos", "first"))
        s["features"] = fs
        rows.append(s.reset_index())
    return pd.concat(rows, ignore_index=True)


def h3_shap(longi: pd.DataFrame, cfg: dict, top=15) -> pd.DataFrame:
    import shap
    X, y = longi.drop(columns=ID_COLS), longi["label"]
    pipe = models(X, cfg["random_state"])["lightgbm"]
    pipe.fit(X, y)
    Xt = pipe.named_steps["prep"].transform(X)
    names = [n.split("__", 1)[-1] for n in pipe.named_steps["prep"].get_feature_names_out()]
    sv = shap.TreeExplainer(pipe.named_steps["clf"]).shap_values(Xt)
    sv = sv[1] if isinstance(sv, list) else sv
    imp = pd.DataFrame({"feature": names, "mean_abs_shap": np.abs(sv).mean(0)})
    hist_cols = set(longi.columns) - set(ID_COLS) - set(_iso_cols(longi))
    imp["type"] = np.where(imp["feature"].str.replace("missingindicator_", "").isin(hist_cols),
                           "history (cross-visit)", "index admission")
    return imp.sort_values("mean_abs_shap", ascending=False).head(top).reset_index(drop=True)


def _iso_cols(longi):
    hist_prefix = ("n_prior", "prior_", "days_since", "n_dx_cumulative", "slope_", "mean_")
    return [c for c in longi.columns if not c.startswith(hist_prefix)]
