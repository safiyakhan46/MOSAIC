"""Train the same two models on both feature sets and compare (H1).

Patient-level split: no patient appears in both train and test.
Metrics: AUPRC (primary), AUROC, Brier, with bootstrap 95% CIs.
"""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    from lightgbm import LGBMClassifier
except ImportError:  # fallback keeps the pipeline runnable
    LGBMClassifier = None
from sklearn.ensemble import HistGradientBoostingClassifier

ID_COLS = ["subject_id", "hadm_id", "label"]
CAT_COLS = ["gender", "admission_type"]


def split(df: pd.DataFrame, cfg: dict):
    gss = GroupShuffleSplit(n_splits=1, test_size=cfg["test_size"],
                            random_state=cfg["random_state"])
    tr, te = next(gss.split(df, df["label"], groups=df["subject_id"]))
    return df["subject_id"].iloc[tr].unique(), df["subject_id"].iloc[te].unique()


def _preprocess(X: pd.DataFrame, scale: bool) -> ColumnTransformer:
    num = [c for c in X.columns if c not in CAT_COLS]
    num_steps = [("impute", SimpleImputer(strategy="median", add_indicator=True))]
    if scale:
        num_steps.append(("scale", StandardScaler()))
    return ColumnTransformer([
        ("num", Pipeline(num_steps), num),
        ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                          ("onehot", OneHotEncoder(handle_unknown="ignore"))]), CAT_COLS),
    ])


def models(X: pd.DataFrame, seed: int) -> dict:
    gbm = (LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=15,
                          class_weight="balanced", random_state=seed, verbose=-1)
           if LGBMClassifier else HistGradientBoostingClassifier(random_state=seed))
    return {
        "logreg": Pipeline([("prep", _preprocess(X, True)),
                            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced"))]),
        "lightgbm": Pipeline([("prep", _preprocess(X, False)), ("clf", gbm)]),
    }


def _metrics(y, p) -> dict:
    if len(np.unique(y)) < 2:
        return {"auprc": np.nan, "auroc": np.nan, "brier": brier_score_loss(y, p)}
    return {"auprc": average_precision_score(y, p),
            "auroc": roc_auc_score(y, p),
            "brier": brier_score_loss(y, p)}


def bootstrap_ci(y, p, n=1000, seed=0) -> dict:
    rng = np.random.default_rng(seed)
    y, p = np.asarray(y), np.asarray(p)
    vals = {"auprc": [], "auroc": []}
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) < 2:
            continue
        vals["auprc"].append(average_precision_score(y[idx], p[idx]))
        vals["auroc"].append(roc_auc_score(y[idx], p[idx]))
    out = {}
    for k, v in vals.items():
        out[f"{k}_lo"] = np.percentile(v, 2.5) if v else np.nan
        out[f"{k}_hi"] = np.percentile(v, 97.5) if v else np.nan
    return out


def compare(iso: pd.DataFrame, longi: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    train_ids, test_ids = split(iso, cfg)
    rows = []
    for set_name, df in [("isolated", iso), ("longitudinal", longi)]:
        tr = df[df["subject_id"].isin(train_ids)]
        te = df[df["subject_id"].isin(test_ids)]
        Xtr, ytr = tr.drop(columns=ID_COLS), tr["label"]
        Xte, yte = te.drop(columns=ID_COLS), te["label"]
        if ytr.nunique() < 2:
            print(f"[warn] {set_name}: training set has one class only — need more data.")
            continue
        for m_name, model in models(Xtr, cfg["random_state"]).items():
            model.fit(Xtr, ytr)
            p = model.predict_proba(Xte)[:, 1]
            rows.append({"features": set_name, "model": m_name,
                         "n_test": len(yte), "pos_rate": yte.mean(),
                         **_metrics(yte, p), **bootstrap_ci(yte, p)})
    return pd.DataFrame(rows)
