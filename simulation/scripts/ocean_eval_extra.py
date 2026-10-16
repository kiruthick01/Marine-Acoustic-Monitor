"""
Follow-up analyses on the trained classifiers (loads weights saved by ocean_classify.py):

 1. zero-shot evaluation on the FULL held-out runs (the 25% adaptation slice is only excluded when a model uses it),
    which is where rare events such as airgun campaigns are evaluable
 2. SNR-stratified behaviour: vessel recall / vessel-type accuracy / airgun recall vs. audibility
 3. storm AP broken down by physical cause (wind, rain, runoff)
 4. few-label fine-tuning from a WEAKER pre-training (A-only, v2 inputs): when does fine-tuning actually matter?
 5. Bayes-optimal bound on cargo-vs-tanker from the true blade rate alone (how much is physically separable)

  python simulation/scripts/ocean_eval_extra.py
"""

import copy
import json
import os
import sys

import numpy as np
import torch
from sklearn.metrics import average_precision_score

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import ocean_classify as oc  # noqa: E402
from simulation.ocean.dataset import LABELS, OUT_DIR  # noqa: E402
from simulation.ocean.models import Norm, predict, train_net  # noqa: E402

RUNS = ["harbor_shelf_s2", "harbor_shelf_s3", "open_coast_s10", "open_coast_s11", "offshore_shelf_s40", "offshore_shelf_s41"]


def load_variant(tag):
    v = oc.Variant(tag, None, 24)
    ck = torch.load(os.path.join(OUT_DIR, "models", f"cnn_{tag}.pt"), weights_only=False)
    v.norm = Norm.from_state(ck["norm"])
    v.net = v.fresh()
    v.net.load_state_dict(ck["state"])
    v.net.eval()
    return v


def ap(y, p):
    return float(average_precision_score(y, p)) if 0 < y.sum() < len(y) else float("nan")


def recall_at_fpr(y, p, fpr=0.01):
    neg = ~y
    if y.sum() == 0 or neg.sum() < 50:
        return float("nan")
    return float((p[y] > np.quantile(p[neg], 1 - fpr)).mean())


def main():
    out = {}
    runs = {n: oc.Run(n) for n in RUNS}
    full = {n: r.cut(0, 1) for n, r in runs.items()}
    variants = {t: load_variant(t) for t in ("Aonly-v1", "Aonly-v2", "multi-v1", "multi-v2")}

    # ---- 1. zero-shot on full runs
    P = {}
    out["full_run_zero_shot"] = {}
    for tag, v in variants.items():
        out["full_run_zero_shot"][tag] = {}
        for n, sl in full.items():
            Pn, VC = predict(v.net, v.data(sl))
            P[(tag, n)] = (Pn, VC)
            out["full_run_zero_shot"][tag][n] = {
                "mAP": oc.eval_scores(sl, Pn, VC, np.full(len(LABELS), 0.5))["mAP"],
                "airgun_ap": ap(sl.df["is_airgun"].values, Pn[:, LABELS.index("is_airgun")]),
                "airgun_n": int(sl.df["is_airgun"].sum()), "fault_ap": ap(sl.df["fault_any"].values, Pn[:, 6])}
        print(tag, {n: (round(d["airgun_ap"], 2), d["airgun_n"]) for n, d in out["full_run_zero_shot"][tag].items()}, flush=True)

    # ---- 2. SNR strata, main model
    out["snr_strata"] = {}
    main = "multi-v2"
    for kind, col, lab, edges in (("vessel", "snr_vessel_db", "is_vessel", [10, 15, 20, 30, 80]),
                                 ("airgun", "snr_airgun_db", "is_airgun", [10, 12, 14, 30])):
        j = LABELS.index(lab)
        rows = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            n_tot = n_hit = n_type = n_type_ok = 0
            for n, sl in full.items():
                Pn, VC = P[(main, n)]
                df = sl.df
                neg = ~df[lab].values
                thr = np.quantile(Pn[neg, j], 0.99)
                m = df[lab].values & (df[col].values >= lo) & (df[col].values < hi)
                n_tot += int(m.sum())
                n_hit += int((Pn[m, j] > thr).sum())
                if kind == "vessel":
                    vc = oc.vclass_idx(df)
                    mm = m & (vc >= 0)
                    n_type += int(mm.sum())
                    n_type_ok += int((VC[mm].argmax(1) == vc[mm]).sum())
            row = {"snr_db": [lo, hi], "n": n_tot, "recall@1%FPR": n_hit / max(n_tot, 1)}
            if kind == "vessel":
                row["vessel_type_acc"] = n_type_ok / max(n_type, 1)
            rows.append(row)
        out["snr_strata"][kind] = rows
        print(kind, rows, flush=True)

    # ---- 3. storm by cause
    j = LABELS.index("storm")
    causes = {"wind>10.5": lambda d: d["wind_ms"].values > 10.5, "rain>2": lambda d: d["rain_mm_h"].values > 2.0,
              "runoff>0.5": lambda d: d["runoff_index"].values > 0.5}
    out["storm_by_cause"] = {}
    for n, sl in full.items():
        Pn, _ = P[(main, n)]
        d = sl.df
        neg = ~d["storm"].values
        thr = np.quantile(Pn[neg, j], 0.95)
        out["storm_by_cause"][n] = {c: {"n": int(f(d).sum()), "recall@5%FPR": float((Pn[f(d), j] > thr).mean()) if f(d).sum() else None}
                                    for c, f in causes.items()}
        out["storm_by_cause"][n]["only_runoff"] = {
            "n": int((causes["runoff>0.5"](d) & ~causes["wind>10.5"](d) & ~causes["rain>2"](d)).sum()),
            "recall@5%FPR": float((Pn[causes["runoff>0.5"](d) & ~causes["wind>10.5"](d) & ~causes["rain>2"](d), j] > thr).mean())
            if (causes["runoff>0.5"](d) & ~causes["wind>10.5"](d) & ~causes["rain>2"](d)).sum() else None}
    print("storm causes", json.dumps(out["storm_by_cause"]["open_coast_s11"]), flush=True)

    # ---- 4. fine-tuning from a weaker (A-only) pre-training
    out["ft_from_Aonly"] = {}
    v = variants["Aonly-v2"]
    base = copy.deepcopy(v.net.state_dict())
    for site, (s1, s2) in oc.HELD_OUT.items():
        r1, r2 = runs[s1], runs[s2]
        ft, val = r1.cut(0.0, 0.15), r1.cut(0.15, 0.25)
        t1, t2 = r1.cut(0.25, 1.0), r2.cut(0.25, 1.0)
        curve = {}
        for n in (0, 108, 216, 432, 648):
            rows = []
            for sd in range(3 if n else 1):
                net = v.fresh()
                net.load_state_dict(base)
                if n:
                    net, _, _ = train_net(net, v.data(ft.head(n)), v.data(val), 14, 1e-3, seed=300 + sd)
                row = {}
                for k, sl in (("season1", t1), ("season2", t2)):
                    Pn, VC = predict(net, v.data(sl))
                    sc = oc.eval_scores(sl, Pn, VC, np.full(len(LABELS), 0.5))
                    row[k] = {"mAP": sc["mAP"], "storm_ap": sc["ap"]["storm"], "vessel_type_acc": sc.get("vessel_type_acc", np.nan)}
                rows.append(row)
            curve[n] = {k: {m: float(np.nanmean([r[k][m] for r in rows])) for m in ("mAP", "storm_ap", "vessel_type_acc")}
                        | {"mAP_std": float(np.std([r[k]["mAP"] for r in rows]))} for k in ("season1", "season2")}
            print(f"FT from A-only, {site} n={n}: s1 {curve[n]['season1']['mAP']:.3f} s2 {curve[n]['season2']['mAP']:.3f} "
                  f"(storm AP s2 {curve[n]['season2']['storm_ap']:.2f})", flush=True)
        out["ft_from_Aonly"][site] = curve

    # ---- 5. Bayes bound on cargo vs tanker from true blade rate
    f0, cls = [], []
    for n in RUNS:
        meta = json.load(open(os.path.join(OUT_DIR, n, "events.json")))
        for p in meta["passages"]:
            if p["cls"] in ("cargo", "tanker"):
                f0.append(p["shaft_hz"] * p["blades"])
                cls.append(p["cls"] == "cargo")
    f0, cls = np.array(f0), np.array(cls)
    bins = np.linspace(3, 11, 33)
    idx = np.digitize(f0, bins)
    correct = 0
    for b in np.unique(idx):
        m = idx == b
        correct += max(cls[m].sum(), (~cls[m]).sum())
    out["cargo_tanker_bayes_bound_true_blade_rate_only"] = {"accuracy": float(correct / len(f0)), "n_passages": int(len(f0)),
                                                              "cargo_share": float(cls.mean())}
    print("Bayes bound", out["cargo_tanker_bayes_bound_true_blade_rate_only"], flush=True)

    with open(os.path.join(OUT_DIR, "extra_results.json"), "w") as f:
        json.dump(out, f, indent=1, default=float)
    print("done")


if __name__ == "__main__":
    main()
