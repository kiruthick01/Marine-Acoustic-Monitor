"""
Stage 2 (unsupervised anomaly detection) on the physics-based ocean dataset.

Protocol (mirrors the deployment story in docs/ml-pipeline.md):
  * every run is calibrated on its OWN first `--cal-days` days (no labels used),
  * the detector then scores all later windows,
  * hyper-parameters / threshold quantiles are tuned on a DEV run only
    (`--dev-run`), and all reported numbers come from the other, unseen runs
    (different weather, different season, different site).

Target "anomaly" = audible vessel | seismic airgun | storm/runoff | instrument fault.
Biology is NOT an anomaly here (it is the Stage-3 classification target).

  python simulation/scripts/ocean_detect.py
"""

import argparse
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd
import torch
from sklearn.covariance import LedoitWolf
from sklearn.ensemble import HistGradientBoostingClassifier, IsolationForest
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from simulation.ocean.dataset import (ACO_EXT, ACO_OLD, CTX, ENV_RAW, OUT_DIR, list_runs, load_run,  # noqa: E402
                                      old_env_cols)
from simulation.pipeline.anomaly_detection import BaselineAnomalyDetector  # noqa: E402

warnings.filterwarnings("ignore")
torch.set_num_threads(max(os.cpu_count() or 4, 4))
TYPES = {"vessel": "is_vessel", "storm": "storm", "airgun": "is_airgun", "fault": "fault_any"}


# ----------------------------------------------------------------- scalers
class RobustZ:
    """Median / scaled-IQR standardiser fitted on the calibration windows."""

    def fit(self, X):
        self.med = np.median(X, axis=0)
        q75, q25 = np.percentile(X, [75, 25], axis=0)
        sd = X.std(axis=0)
        self.scale = np.maximum((q75 - q25) / 1.349, 0.25 * sd + 1e-9)
        return self

    def transform(self, X, clip=12.0):
        return np.clip((X - self.med) / self.scale, -clip, clip)


# --------------------------------------------------------------- detectors
# Each detector: fit(cal_df) -> self ; score(df) -> 1-D array (higher = more anomalous)
class OldBaseline:
    """Existing project detector on the existing 34+8 features (threshold handled by caller)."""
    name = "M0 baseline IF (34 acoustic + 8 env)"

    def fit(self, cal):
        self.cols = ACO_OLD + old_env_cols(cal)
        self.det = BaselineAnomalyDetector(random_state=0).fit(cal[self.cols])
        return self

    def score(self, df):
        return -self.det._model.decision_function(df[self.cols].values)


class IFExt:
    name = "M1 IF on extended calibrated features"

    def __init__(self, n_estimators=300, max_features=1.0):
        self.kw = dict(n_estimators=n_estimators, max_features=max_features)

    def fit(self, cal):
        self.cols = ACO_EXT + ENV_RAW + CTX
        self.sc = RobustZ().fit(cal[self.cols].values)
        self.m = IsolationForest(random_state=0, n_jobs=-1, **self.kw).fit(self.sc.transform(cal[self.cols].values))
        return self

    def score(self, df):
        return -self.m.decision_function(self.sc.transform(df[self.cols].values))


class Balanced:
    """Modality-balanced: separate acoustic and environmental scorers, fused by max of calibration z-scores.
    Fixes the subspace-dilution weakness diagnosed in docs/ml-pipeline.md (34 vs 8 dims)."""
    name = "M2 modality-balanced IF fusion"

    def fit(self, cal):
        self.ac, self.ec = ACO_EXT + CTX, ENV_RAW + CTX
        self.sa = RobustZ().fit(cal[self.ac].values)
        self.se = RobustZ().fit(cal[self.ec].values)
        self.ma = IsolationForest(n_estimators=300, random_state=0, n_jobs=-1).fit(self.sa.transform(cal[self.ac].values))
        self.me = IsolationForest(n_estimators=300, random_state=0, n_jobs=-1).fit(self.se.transform(cal[self.ec].values))
        a, e = self._raw(cal)
        self.mu = (a.mean(), e.mean())
        self.sd = (a.std() + 1e-9, e.std() + 1e-9)
        return self

    def _raw(self, df):
        return (-self.ma.decision_function(self.sa.transform(df[self.ac].values)),
                -self.me.decision_function(self.se.transform(df[self.ec].values)))

    def score(self, df):
        a, e = self._raw(df)
        return np.maximum((a - self.mu[0]) / self.sd[0], (e - self.mu[1]) / self.sd[1])


class Mahalanobis:
    name = "M3 Mahalanobis (Ledoit-Wolf)"

    def fit(self, cal):
        self.cols = ACO_EXT + ENV_RAW + CTX
        self.sc = RobustZ().fit(cal[self.cols].values)
        self.lw = LedoitWolf().fit(self.sc.transform(cal[self.cols].values))
        return self

    def score(self, df):
        return self.lw.mahalanobis(self.sc.transform(df[self.cols].values))


class AutoEncoder:
    name = "M4 autoencoder"

    def __init__(self, hidden=64, bottleneck=8, wd=1e-3, epochs=250, seed=0):
        self.h, self.b, self.wd, self.epochs, self.seed = hidden, bottleneck, wd, epochs, seed

    def fit(self, cal):
        torch.manual_seed(self.seed)
        self.cols = ACO_EXT + ENV_RAW + CTX
        self.sc = RobustZ().fit(cal[self.cols].values)
        X = torch.tensor(self.sc.transform(cal[self.cols].values), dtype=torch.float32)
        d = X.shape[1]
        self.net = torch.nn.Sequential(
            torch.nn.Linear(d, self.h), torch.nn.GELU(), torch.nn.Linear(self.h, self.b), torch.nn.GELU(),
            torch.nn.Linear(self.b, self.h), torch.nn.GELU(), torch.nn.Linear(self.h, d))
        opt = torch.optim.AdamW(self.net.parameters(), lr=2e-3, weight_decay=self.wd)
        n = len(X)
        for ep in range(self.epochs):
            perm = torch.randperm(n)
            for s in range(0, n, 128):
                xb = X[perm[s:s + 128]]
                noisy = xb + 0.1 * torch.randn_like(xb)
                loss = torch.nn.functional.smooth_l1_loss(self.net(noisy), xb)
                opt.zero_grad()
                loss.backward()
                opt.step()
        return self

    def score(self, df):
        X = torch.tensor(self.sc.transform(df[self.cols].values), dtype=torch.float32)
        with torch.no_grad():
            err = torch.nn.functional.smooth_l1_loss(self.net(X), X, reduction="none")
        return err.mean(1).numpy()


class Trimmed:
    """Label-free self-cleaning of a contaminated calibration period: fit, drop the highest-scoring
    `trim` fraction of calibration windows (likely events), refit. Repeats `rounds` times."""

    def __init__(self, make, trim=0.2, rounds=2, name="trimmed"):
        self.make, self.trim, self.rounds, self.name = make, trim, rounds, name

    def fit(self, cal):
        cur = cal
        for _ in range(self.rounds):
            self.det = self.make().fit(cur)
            s = self.det.score(cal)
            cur = cal[s <= np.quantile(s, 1 - self.trim)]
        self.det = self.make().fit(cur)
        return self

    def score(self, df):
        return self.det.score(df)


class Ensemble:
    """Rank-free fusion: each member's score z-normalised on calibration, then averaged (max of group means)."""
    name = "M5 ensemble (M2+M3+M4)"

    def __init__(self, ae_kw=None):
        self.members = [Balanced(), Mahalanobis(), AutoEncoder(**(ae_kw or {}))]

    def fit(self, cal):
        self.stats = []
        for m in self.members:
            m.fit(cal)
            s = m.score(cal)
            lo, hi = np.percentile(s, [50, 95])
            self.stats.append((lo, max(hi - lo, 1e-9)))
        return self

    def score(self, df):
        z = [(m.score(df) - lo) / sc for m, (lo, sc) in zip(self.members, self.stats)]
        return np.mean(z, axis=0)


# ---------------------------------------------------------------- evaluation
def thresholds_from_cal(cal_scores, q):
    return float(np.quantile(cal_scores, q))


def eval_run(det, df, cal_n, q, clean_cal=False):
    cal = df.iloc[:cal_n]
    if clean_cal:
        cal = cal[~cal["anomaly"]]
    det.fit(cal)
    s_cal = det.score(cal)
    s = det.score(df.iloc[cal_n:])
    y = df["anomaly"].values[cal_n:].astype(int)
    tau = thresholds_from_cal(s_cal, q)
    pred = (s > tau).astype(int)
    res = dict(auroc=roc_auc_score(y, s), auprc=average_precision_score(y, s), base_rate=float(y.mean()),
               precision=precision_score(y, pred, zero_division=0), recall=recall_score(y, pred, zero_division=0),
               f1=f1_score(y, pred, zero_division=0), flag_rate=float(pred.mean()))
    for k, col in TYPES.items():
        m = df[col].values[cal_n:]
        res[f"recall_{k}"] = float(pred[m].mean()) if m.any() else np.nan
        res[f"auroc_{k}"] = roc_auc_score(m, s) if 0 < m.sum() < len(m) else np.nan
    return res, s, pred


def rolling_recal(make_det, df, cal_n, q, refit_every=1008, window=1008):
    """Weekly refit on the previous week's windows the detector itself did NOT flag (no labels)."""
    det = make_det().fit(df.iloc[:cal_n])
    tau = thresholds_from_cal(det.score(df.iloc[:cal_n]), q)
    scores, preds = [], []
    i = cal_n
    n = len(df)
    while i < n:
        j = min(i + refit_every, n)
        chunk = df.iloc[i:j]
        s = det.score(chunk)
        p = s > tau
        scores.append(s)
        preds.append(p)
        ctx_df = df.iloc[max(0, j - window):j]
        flagged = np.concatenate(preds)[-len(ctx_df):] if len(np.concatenate(preds)) >= len(ctx_df) else np.zeros(len(ctx_df), bool)
        keep = ctx_df[~flagged[-len(ctx_df):]]
        if len(keep) > 150 and j < n:
            det = make_det().fit(keep)
            tau = thresholds_from_cal(det.score(keep), q)
        i = j
    s = np.concatenate(scores)
    pred = np.concatenate(preds).astype(int)
    y = df["anomaly"].values[cal_n:].astype(int)
    return dict(auroc=roc_auc_score(y, s), auprc=average_precision_score(y, s),
                precision=precision_score(y, pred, zero_division=0), recall=recall_score(y, pred, zero_division=0),
                f1=f1_score(y, pred, zero_division=0))


def supervised_ceiling(runs, test_name, cal_n):
    """Labelled upper bound: gradient boosting trained on OTHER runs' labelled windows (not deployable at cold start)."""
    cols = ACO_EXT + ENV_RAW + CTX
    tr = pd.concat([d for n, d in runs.items() if n != test_name and d["site"].iloc[0] == runs[test_name]["site"].iloc[0]])
    te = runs[test_name].iloc[cal_n:]
    clf = HistGradientBoostingClassifier(max_iter=250, learning_rate=0.08, random_state=0).fit(tr[cols], tr["anomaly"])
    p = clf.predict_proba(te[cols])[:, 1]
    y = te["anomaly"].astype(int).values
    pred = (p > 0.5).astype(int)
    return dict(auroc=roc_auc_score(y, p), auprc=average_precision_score(y, p), f1=f1_score(y, pred),
                precision=precision_score(y, pred, zero_division=0), recall=recall_score(y, pred, zero_division=0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cal-days", type=float, default=3.0)
    ap.add_argument("--dev-run", default="harbor_shelf_s2")
    ap.add_argument("--interval-min", type=float, default=10.0)
    a = ap.parse_args()
    cal_n = int(a.cal_days * 1440 / a.interval_min)

    runs = {n: load_run(n)[0] for n in list_runs()}
    test_names = [n for n in runs if n != a.dev_run]
    print(f"runs: {list(runs)} | dev={a.dev_run} | calibration={cal_n} windows")
    for n, d in runs.items():
        print(f"  {n}: anomaly {d['anomaly'].mean():.1%}  vessel {d['is_vessel'].mean():.1%} storm {d['storm'].mean():.1%} "
              f"airgun {d['is_airgun'].mean():.1%} fault {d['fault_any'].mean():.1%}")

    factories = {
        "M0": OldBaseline, "M1": IFExt, "M2": Balanced, "M3": Mahalanobis, "M4": AutoEncoder, "M5": Ensemble,
        "M6": lambda: Trimmed(IFExt, 0.25, 2, "M6 self-cleaned IF (trimmed refit)"),
        "M7": lambda: Trimmed(Mahalanobis, 0.25, 2, "M7 self-cleaned Mahalanobis"),
    }
    qs = [0.8, 0.85, 0.9, 0.95, 0.97, 0.99, 1.0]

    # ---- tune: AE hyper-parameters + per-method threshold quantile on the dev run only
    dev = runs[a.dev_run]
    ae_grid = [dict(hidden=h, bottleneck=b, wd=w) for h in (32, 64) for b in (4, 8) for w in (1e-3, 1e-2)]
    best_ae, best = None, -1
    for kw in ae_grid:
        r, _, _ = eval_run(AutoEncoder(epochs=120, **kw), dev, cal_n, 0.99)
        print("  AE tune", kw, f"auprc={r['auprc']:.3f}")
        if r["auprc"] > best:
            best, best_ae = r["auprc"], kw
    print("best AE:", best_ae)
    factories["M4"] = lambda: AutoEncoder(epochs=200, **best_ae)
    factories["M5"] = lambda: Ensemble(ae_kw=dict(epochs=200, **best_ae))
    factories["M8"] = lambda: Trimmed(lambda: AutoEncoder(epochs=200, **best_ae), 0.25, 2, "M8 self-cleaned autoencoder")

    best_q = {}
    dev_cache = {}
    for k, f in factories.items():
        det = f()
        r, s, _ = eval_run(det, dev, cal_n, 1.0)
        dev_cache[k] = (det, s)
        s_cal = det.score(dev.iloc[:cal_n])
        y = dev["anomaly"].values[cal_n:].astype(int)
        f1s = {q: f1_score(y, (s > thresholds_from_cal(s_cal, q)).astype(int), zero_division=0) for q in qs}
        best_q[k] = max(f1s, key=f1s.get)
        print(f"  {k} dev F1 by q: " + ", ".join(f"{q}:{v:.2f}" for q, v in f1s.items()) + f" -> q={best_q[k]}")

    # ---- test on unseen runs
    results = {}
    for k, f in factories.items():
        per = {}
        for n in test_names:
            r, _, _ = eval_run(f(), runs[n], cal_n, best_q[k])
            per[n] = r
        results[k] = per
        mean = pd.DataFrame(per).T.mean(numeric_only=True)
        print(f"{f().name:44s} AUROC {mean['auroc']:.3f} AUPRC {mean['auprc']:.3f} P {mean['precision']:.3f} "
              f"R {mean['recall']:.3f} F1 {mean['f1']:.3f} | recall vessel {mean['recall_vessel']:.2f} storm {mean['recall_storm']:.2f} "
              f"airgun {mean['recall_airgun']:.2f} fault {mean['recall_fault']:.2f}")

    # ---- calibration contamination ablation (best unsupervised method)
    best_k = max(results, key=lambda k: pd.DataFrame(results[k]).T["auprc"].mean())
    clean = {n: eval_run(factories[best_k](), runs[n], cal_n, best_q[best_k], clean_cal=True)[0] for n in test_names}
    print(f"\nbest unsupervised: {best_k}; clean-calibration ablation F1 {pd.DataFrame(clean).T['f1'].mean():.3f} "
          f"vs contaminated {pd.DataFrame(results[best_k]).T['f1'].mean():.3f}")

    # ---- rolling recalibration vs static (long runs only)
    roll = {n: rolling_recal(factories[best_k], runs[n], cal_n, best_q[best_k]) for n in test_names}
    print(f"rolling weekly recalibration F1 {pd.DataFrame(roll).T['f1'].mean():.3f} AUPRC {pd.DataFrame(roll).T['auprc'].mean():.3f}")

    # ---- supervised ceiling (same-site, leave-one-run-out)
    sup = {n: supervised_ceiling(runs, n, cal_n) for n in test_names}
    print(f"supervised ceiling (labelled, other runs): AUROC {pd.DataFrame(sup).T['auroc'].mean():.3f} "
          f"AUPRC {pd.DataFrame(sup).T['auprc'].mean():.3f} F1 {pd.DataFrame(sup).T['f1'].mean():.3f}")

    out = dict(cal_windows=cal_n, dev_run=a.dev_run, test_runs=test_names, best_ae=best_ae, best_q={k: float(v) for k, v in best_q.items()},
               methods={k: factories[k]().name for k in factories}, results=results, best_unsupervised=best_k,
               clean_calibration=clean, rolling_recalibration=roll, supervised_ceiling=sup)
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "detection_results.json"), "w") as f:
        json.dump(out, f, indent=1, default=float)

    # save the scores of the best detector on test runs for rendering
    scores = {}
    for n in test_names:
        r, s, pred = eval_run(factories[best_k](), runs[n], cal_n, best_q[best_k])
        scores[n] = dict(score=s.tolist(), pred=pred.tolist())
    with open(os.path.join(OUT_DIR, "detection_scores.json"), "w") as f:
        json.dump(scores, f)


if __name__ == "__main__":
    main()
