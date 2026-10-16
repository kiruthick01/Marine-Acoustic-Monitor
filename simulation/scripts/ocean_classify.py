"""
Stage 3: supervised multi-label classification, domain generalisation and few-label adaptation on the ocean corpus.

Labels per window: vessel, dolphin, whale, fish chorus, seismic airgun, storm, instrument fault (+ 5-way vessel type).

Corpus (see ocean_generate.py)
  train   sites A, C, D (+ event-rich runs), first 80% of each run     val  last 20% (early stop, thresholds, calibration)
  test    A s2 (unseen weather), A s3 (unseen season), and two fully HELD-OUT sites, B and E, each in two seasons.
          Held-out windows used for adaptation (first 25% of every B/E run) are never in the test slice.

Model grid (ablation)   inputs {v1: mel + env, v2: + sensor-health context + narrowband LOFAR/comb}  x
                        training {A only, multi-site A+C+D}; GB baselines on the matching tabular features.
Domain adaptation       zero-shot, AdaBN (unlabeled BN-statistics re-estimation), few-label fine-tuning learning curve,
                        and two-season re-adaptation; scratch baseline trained on the same labelled windows.

  python simulation/scripts/ocean_classify.py                # train + evaluate everything (~2-3 h on 16 cores)
  python simulation/scripts/ocean_classify.py --skip-train   # reuse saved weights in output/ocean/models
"""

import argparse
import copy
import json
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from simulation.ocean.dataset import (ACO_EXT, COMB, CTX, ENV_RAW, LABELS, OUT_DIR, TEMPORAL,  # noqa: E402
                                      VESSEL_CLASSES, load_lofar, load_run)
from simulation.ocean.models import (Data, Net, Norm, adabn, ece, fit_temperature, make_data,  # noqa: E402
                                     predict, train_net)

warnings.filterwarnings("ignore")
torch.set_num_threads(max(os.cpu_count() or 4, 4))

TAB_V1 = ACO_EXT + ENV_RAW + CTX
TAB_V2 = TAB_V1 + TEMPORAL + COMB
VEC_V1 = ENV_RAW + CTX
VEC_V2 = ENV_RAW + CTX + TEMPORAL + COMB
MODELS_DIR = os.path.join(OUT_DIR, "models")

TRAIN_MULTI = ["harbor_shelf_s0", "harbor_shelf_s1", "estuary_bay_s20", "estuary_bay_s21",
               "strait_channel_s30", "strait_channel_s31",
               "rich_harbor_s4", "rich_harbor_s5", "rich_harbor_s6", "rich_estuary_s7", "rich_strait_s8"]
TRAIN_A = ["harbor_shelf_s0", "harbor_shelf_s1", "rich_harbor_s4", "rich_harbor_s5", "rich_harbor_s6"]
HELD_OUT = {"B": ("open_coast_s10", "open_coast_s11"), "E": ("offshore_shelf_s40", "offshore_shelf_s41")}


# ------------------------------------------------------------------ helpers
def vclass_idx(df):
    m = {c: i for i, c in enumerate(VESSEL_CLASSES)}
    return np.array([m.get(c, -1) if isinstance(c, str) else -1 for c in df["vessel_class"]])


class Run:
    """One simulated run with chronological slicing."""

    def __init__(self, name):
        self.name = name
        self.df, self.mel = load_run(name)
        self.lofar = load_lofar(name)

    def cut(self, a, b):
        n = len(self.df)
        sl = slice(int(round(a * n)), int(round(b * n)))
        return Slice(self.df.iloc[sl], self.mel[sl], self.lofar[sl])


class Slice:
    def __init__(self, df, mel, lofar):
        self.df, self.mel, self.lofar = df, mel, lofar

    def __len__(self):
        return len(self.df)

    @staticmethod
    def concat(parts):
        return Slice(pd.concat([p.df for p in parts]), np.concatenate([p.mel for p in parts]),
                     np.concatenate([p.lofar for p in parts]))

    def every(self, k):
        return Slice(self.df.iloc[::k], self.mel[::k], self.lofar[::k])

    def head(self, n):
        return Slice(self.df.iloc[:n], self.mel[:n], self.lofar[:n])


def eval_scores(sl, P, VC, thr):
    """Threshold-free + thresholded metrics, fault-type recall at 1% FPR, and vessel-type breakdowns."""
    df = sl.df
    Y = df[LABELS].values.astype(int)
    out = {"ap": {}, "f1": {}, "n_pos": {}}
    for j, lab in enumerate(LABELS):
        y = Y[:, j]
        out["n_pos"][lab] = int(y.sum())
        out["ap"][lab] = float(average_precision_score(y, P[:, j])) if 0 < y.sum() < len(y) else float("nan")
        out["f1"][lab] = float(f1_score(y, P[:, j] > thr[j], zero_division=0)) if y.sum() > 0 else float("nan")
    out["mAP"] = float(np.nanmean(list(out["ap"].values())))
    out["macroF1"] = float(np.nanmean(list(out["f1"].values())))
    # fault type recall at a fixed 1% false-positive rate (what a deployed alarm would use)
    j = LABELS.index("fault_any")
    neg = ~df["fault_any"].values
    if neg.sum() > 100:
        t = np.quantile(P[neg, j], 0.99)
        ftype = {"hum": df["fault_audio"].values == "hum", "pops": df["fault_audio"].values == "pops",
                 "dropout": df["fault_audio"].values == "dropout", "stuck_sensor": df["fault_env"].values}
        out["fault_recall@1%FPR"] = {k: (float((P[m, j] > t).mean()) if m.sum() else float("nan"))
                                     for k, m in ftype.items()}
    vc = vclass_idx(df)
    m = vc >= 0
    if VC is not None and m.sum() > 5:
        pred = VC[m].argmax(1)
        out["vessel_type_acc"] = float((pred == vc[m]).mean())
        out["vessel_type_macroF1"] = float(f1_score(vc[m], pred, average="macro", zero_division=0))
        fam = np.array([0, 0, 1, 2, 3])        # merchant (cargo+tanker), ferry, fishing, outboard
        pf = np.stack([VC[m][:, :2].sum(1), VC[m][:, 2], VC[m][:, 3], VC[m][:, 4]], 1).argmax(1)
        out["vessel_family_acc"] = float((pf == fam[vc[m]]).mean())
        ct = m & (vc <= 1)
        if ct.sum() > 5 and len(set(vc[ct])) == 2:
            s = VC[ct][:, 0] / (VC[ct][:, 0] + VC[ct][:, 1] + 1e-9)
            out["cargo_vs_tanker_acc"] = float(((s > 0.5) == (vc[ct] == 0)).mean())
            out["cargo_vs_tanker_auc"] = float(roc_auc_score(vc[ct] == 0, s))
        loud = m & (df["snr_vessel_db"].values >= 20)
        if loud.sum() > 5:
            out["vessel_type_acc_snr>=20dB"] = float((VC[loud].argmax(1) == vc[loud]).mean())
    return out


def best_thresholds(Y, P):
    thr = np.full(Y.shape[1], 0.5)
    for j in range(Y.shape[1]):
        if Y[:, j].sum() == 0:
            continue
        cands = np.linspace(0.05, 0.95, 19)
        thr[j] = cands[int(np.argmax([f1_score(Y[:, j], P[:, j] > t, zero_division=0) for t in cands]))]
    return thr


# ------------------------------------------------------------------ gradient boosting
def fit_gb(tr, va, cols, trials=6, seed=0, sample_weight=None):
    rng = np.random.default_rng(seed)
    Xtr, Xva = tr[cols].values, (va[cols].values if va is not None else None)
    models = []
    for lab in LABELS:
        y = tr[lab].values.astype(int)
        if y.sum() < 3:
            models.append(None)
            continue
        yv = va[lab].values.astype(int) if va is not None else None
        best, bm = -1, None
        for _ in range(trials if va is not None else 1):
            kw = dict(learning_rate=float(rng.choice([0.04, 0.08, 0.15])), max_leaf_nodes=int(rng.choice([8, 15, 31])),
                      l2_regularization=float(rng.choice([0.0, 0.5, 2.0])), max_iter=int(rng.choice([150, 300])),
                      min_samples_leaf=int(rng.choice([10, 20])), random_state=seed)
            m = HistGradientBoostingClassifier(class_weight="balanced", **kw)
            m.fit(Xtr, y, sample_weight=sample_weight)
            sc = average_precision_score(yv, m.predict_proba(Xva)[:, 1]) if (va is not None and yv.sum() > 0) else 0.0
            if sc > best:
                best, bm = sc, m
        models.append(bm)
    return models


def pred_gb(models, df, cols):
    X = df[cols].values
    return np.stack([m.predict_proba(X)[:, 1] if m is not None else np.zeros(len(df)) for m in models], axis=1)


# ------------------------------------------------------------------ CNN variants
def new_net(variant, width):
    v2 = variant.endswith("v2")
    return Net(len(VEC_V2 if v2 else VEC_V1), len(LABELS), len(VESSEL_CLASSES), width, use_lofar=v2)


class Variant:
    """Bundles net + normaliser + the column set for one (inputs, training-corpus) combination."""

    def __init__(self, tag, train_names, width):
        self.tag = tag
        self.v2 = tag.endswith("v2")
        self.cols = VEC_V2 if self.v2 else VEC_V1
        self.width = width
        self.train_names = train_names
        self.net = None
        self.norm = None

    def data(self, sl):
        return make_data(self.norm, sl.df, sl.mel, sl.lofar, self.cols, LABELS, vclass_idx, self.v2)

    def fresh(self):
        return new_net(self.tag, self.width)

    def path(self):
        return os.path.join(MODELS_DIR, f"cnn_{self.tag}.pt")


def split_corpus(runs, names):
    tr = Slice.concat([runs[n].cut(0.0, 0.8) for n in names])
    va = Slice.concat([runs[n].cut(0.8, 1.0) for n in names])
    return tr, va


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=16)
    ap.add_argument("--ft-seeds", type=int, default=3)
    ap.add_argument("--width", type=int, default=24)
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="tiny subsampled run to test the code path")
    ap.add_argument("--neg-frac", type=float, default=0.5)
    a = ap.parse_args()
    t_start = time.time()
    global MODELS_DIR
    if a.smoke:
        MODELS_DIR = os.path.join(OUT_DIR, "models_smoke")
    os.makedirs(MODELS_DIR, exist_ok=True)

    names = sorted(set(TRAIN_MULTI) | {"harbor_shelf_s2", "harbor_shelf_s3"} | {n for v in HELD_OUT.values() for n in v})
    runs = {n: Run(n) for n in names}
    print("loaded", len(runs), "runs", flush=True)

    tests = {"A_unseen_weather": runs["harbor_shelf_s2"].cut(0, 1), "A_unseen_season": runs["harbor_shelf_s3"].cut(0, 1)}
    adapt = {}
    for site, (spring, other) in HELD_OUT.items():
        tests[f"{site}_season1"] = runs[spring].cut(0.25, 1.0)
        tests[f"{site}_season2"] = runs[other].cut(0.25, 1.0)
        adapt[site] = {"s1_ft": runs[spring].cut(0.0, 0.15), "s1_val": runs[spring].cut(0.15, 0.25),
                       "s2_ft": runs[other].cut(0.0, 0.075), "s2_val": runs[other].cut(0.15, 0.25),
                       "s1_unlab": runs[spring].cut(0.0, 0.25), "s2_unlab": runs[other].cut(0.0, 0.25)}
    tr_multi, va_multi = split_corpus(runs, TRAIN_MULTI)
    tr_a, va_a = split_corpus(runs, TRAIN_A)
    if a.smoke:
        tr_multi, va_multi, tr_a, va_a = (x.every(12) for x in (tr_multi, va_multi, tr_a, va_a))
        tests = {k: v.every(6) for k, v in tests.items()}
        adapt = {s: {k: v.every(4) for k, v in d.items()} for s, d in adapt.items()}
    print(f"train multi {len(tr_multi)} (A-only {len(tr_a)}), val {len(va_multi)}", flush=True)
    print("train prevalence: " + ", ".join(f"{l}={tr_multi.df[l].mean():.3f}" for l in LABELS), flush=True)

    R = {"config": vars(a), "tests": {k: {"n": len(v), "prevalence": {l: float(v.df[l].mean()) for l in LABELS},
                                          "n_vessel_typed": int((vclass_idx(v.df) >= 0).sum())} for k, v in tests.items()}}
    preds = {}
    Ytest = {k: v.df[LABELS].values.astype(int) for k, v in tests.items()}
    VCtest = {k: vclass_idx(v.df) for k, v in tests.items()}

    # ------------- gradient boosting baselines
    print("\n== Gradient boosting ==", flush=True)
    gb_cfg = {"GB-v1-Aonly": (tr_a, va_a, TAB_V1), "GB-v1-multi": (tr_multi, va_multi, TAB_V1),
              "GB-v2-multi": (tr_multi, va_multi, TAB_V2)}
    for tag, (tr, va, cols) in gb_cfg.items():
        models = fit_gb(tr.df, va.df, cols)
        thr = best_thresholds(va.df[LABELS].values.astype(int), pred_gb(models, va.df, cols))
        for k, sl in tests.items():
            P = pred_gb(models, sl.df, cols)
            R[f"{tag}/{k}"] = eval_scores(sl, P, None, thr)
            preds[f"{tag}/{k}"] = P
        print(f"  {tag:14s} " + " ".join(f"{k}:{R[f'{tag}/{k}']['mAP']:.3f}" for k in tests), flush=True)

    # ------------- CNN grid
    print("\n== CNN grid (inputs x training corpus) ==", flush=True)
    grid = [Variant("Aonly-v1", TRAIN_A, a.width), Variant("Aonly-v2", TRAIN_A, a.width),
            Variant("multi-v1", TRAIN_MULTI, a.width), Variant("multi-v2", TRAIN_MULTI, a.width)]
    thr_of = {}
    for v in grid:
        tr, va = (tr_a, va_a) if v.train_names is TRAIN_A else (tr_multi, va_multi)
        v.norm = Norm(tr.mel, tr.df[v.cols].values)
        v.net = v.fresh()
        if a.skip_train and os.path.exists(v.path()):
            v.net.load_state_dict(torch.load(v.path(), weights_only=False)["state"])
            print(f"  [{v.tag}] loaded", flush=True)
        else:
            t0 = time.time()
            print(f"  [{v.tag}] params {sum(p.numel() for p in v.net.parameters()):,}", flush=True)
            v.net, hist, best = train_net(v.net, v.data(tr), v.data(va), a.epochs, 2e-3, seed=0, log=True,
                                          neg_frac=a.neg_frac)
            torch.save({"state": v.net.state_dict(), "tag": v.tag, "norm": v.norm.state(), "cols": v.cols, "width": a.width}, v.path())
            R[f"history/{v.tag}"] = hist
            print(f"  [{v.tag}] trained in {(time.time() - t0) / 60:.1f} min, best val mAP {best:.3f}", flush=True)
        Pva, _ = predict(v.net, v.data(va))
        thr_of[v.tag] = best_thresholds(va.df[LABELS].values.astype(int), Pva)
        for k, sl in tests.items():
            d = v.data(sl)
            P, VC = predict(v.net, d)
            R[f"CNN-{v.tag}/{k}"] = eval_scores(sl, P, VC, thr_of[v.tag])
            preds[f"CNN-{v.tag}/{k}"], preds[f"CNN-{v.tag}-vc/{k}"] = P, VC
            # unsupervised adaptation: BN statistics from the first 25% of the target run (no labels)
            if k.startswith(("B_", "E_")):
                site = k[0]
                un = adapt[site]["s1_unlab" if k.endswith("season1") else "s2_unlab"]
                net_ab = adabn(v.net, v.data(un))
                P2, VC2 = predict(net_ab, d)
                R[f"CNN-{v.tag}+AdaBN/{k}"] = eval_scores(sl, P2, VC2, thr_of[v.tag])
                preds[f"CNN-{v.tag}+AdaBN/{k}"], preds[f"CNN-{v.tag}+AdaBN-vc/{k}"] = P2, VC2
        print(f"  {v.tag:10s} " + " ".join(f"{k}:{R[f'CNN-{v.tag}/{k}']['mAP']:.3f}" for k in tests), flush=True)

    # ------------- calibration of the main model (temperature scaling fitted on val)
    main_v = grid[3]
    va_d = main_v.data(va_multi)
    lg_va, _ = predict(main_v.net, va_d, logits=True)
    T = fit_temperature(lg_va, va_multi.df[LABELS].values.astype(int))
    R["calibration"] = {"temperature": T.tolist(), "ece_raw": {}, "ece_scaled": {}}
    for k, sl in tests.items():
        lg, _ = predict(main_v.net, main_v.data(sl), logits=True)
        Pr, Ps = 1 / (1 + np.exp(-lg)), 1 / (1 + np.exp(-lg / T))
        R["calibration"]["ece_raw"][k] = {l: ece(Pr[:, j], Ytest[k][:, j]) for j, l in enumerate(LABELS)}
        R["calibration"]["ece_scaled"][k] = {l: ece(Ps[:, j], Ytest[k][:, j]) for j, l in enumerate(LABELS)}
        preds[f"CNN-multi-v2-cal/{k}"] = Ps

    # ------------- few-label adaptation at the held-out sites
    print("\n== Few-label adaptation (main model: multi-v2) ==", flush=True)
    base_state = copy.deepcopy(main_v.net.state_dict())
    sizes = [0, 54, 108, 216, 432, 648] if not a.smoke else [0, 54]
    R["adaptation"] = {}
    for site in HELD_OUT:
        ad = adapt[site]
        k1, k2 = f"{site}_season1", f"{site}_season2"
        curve = {"sizes": sizes, "season1": {}, "season2": {}}
        for n in sizes:
            runs_s = []
            for sd in range(a.ft_seeds if n > 0 else 1):
                net = main_v.fresh()
                net.load_state_dict(base_state)
                if n > 0:
                    net, _, _ = train_net(net, main_v.data(ad["s1_ft"].head(n)), main_v.data(ad["s1_val"]), 14, 1e-3,
                                          seed=100 + sd)
                Pv, _ = predict(net, main_v.data(ad["s1_val"]))
                thr = best_thresholds(ad["s1_val"].df[LABELS].values.astype(int), Pv) if n > 0 else thr_of[main_v.tag]
                row = {}
                for kk in (k1, k2):
                    P, VC = predict(net, main_v.data(tests[kk]))
                    row[kk] = eval_scores(tests[kk], P, VC, thr)
                    if sd == 0 and n in (0, 648):
                        preds[f"CNN-ft{n}/{kk}"], preds[f"CNN-ft{n}-vc/{kk}"] = P, VC
                runs_s.append(row)
            for kk, key in ((k1, "season1"), (k2, "season2")):
                v = [r[kk]["mAP"] for r in runs_s]
                curve[key][n] = {"mAP_mean": float(np.mean(v)), "mAP_std": float(np.std(v)),
                                 "per_label_ap": {l: float(np.nanmean([r[kk]["ap"][l] for r in runs_s])) for l in LABELS},
                                 "vessel_type_acc": float(np.nanmean([r[kk].get("vessel_type_acc", np.nan) for r in runs_s]))}
            print(f"  {site} n={n:3d}: season1 {curve['season1'][n]['mAP_mean']:.3f}±{curve['season1'][n]['mAP_std']:.3f} "
                  f"season2 {curve['season2'][n]['mAP_mean']:.3f}±{curve['season2'][n]['mAP_std']:.3f}", flush=True)

        # two-season re-adaptation: 324 labelled windows from season 1 + 324 from season 2
        half = min(324, len(ad["s1_ft"]), len(ad["s2_ft"]))
        two_tr = Slice.concat([ad["s1_ft"].head(half), ad["s2_ft"].head(half)])
        two_va = Slice.concat([ad["s1_val"], ad["s2_val"]])
        two, scratch, one = [], [], []
        for sd in range(a.ft_seeds):
            net = main_v.fresh()
            net.load_state_dict(base_state)
            net, _, _ = train_net(net, main_v.data(two_tr), main_v.data(two_va), 14, 1e-3, seed=200 + sd)
            Pv, _ = predict(net, main_v.data(two_va))
            thr = best_thresholds(two_va.df[LABELS].values.astype(int), Pv)
            ns = main_v.fresh()
            ns, _, _ = train_net(ns, main_v.data(ad["s1_ft"]), main_v.data(ad["s1_val"]), 30, 2e-3, seed=200 + sd)
            Pv2, _ = predict(ns, main_v.data(ad["s1_val"]))
            thr2 = best_thresholds(ad["s1_val"].df[LABELS].values.astype(int), Pv2)
            for model, th, store in ((net, thr, two), (ns, thr2, scratch)):
                store.append({kk: eval_scores(tests[kk], *predict(model, main_v.data(tests[kk])), th) for kk in (k1, k2)})
        R["adaptation"][site] = {
            "curve": curve,
            "two_season_648": {kk: {"mAP_mean": float(np.mean([r[kk]["mAP"] for r in two])),
                                    "mAP_std": float(np.std([r[kk]["mAP"] for r in two]))} for kk in (k1, k2)},
            "scratch_648_season1_only": {kk: {"mAP_mean": float(np.mean([r[kk]["mAP"] for r in scratch])),
                                              "mAP_std": float(np.std([r[kk]["mAP"] for r in scratch]))} for kk in (k1, k2)},
        }
        # GB baseline with target windows added
        both = pd.concat([tr_multi.df, ad["s1_ft"].df])
        w = np.concatenate([np.ones(len(tr_multi)), np.full(len(ad["s1_ft"]), 4.0)])
        gbm = fit_gb(both, None, TAB_V2, sample_weight=w)
        R["adaptation"][site]["gb_multi+648"] = {kk: float(eval_scores(tests[kk], pred_gb(gbm, tests[kk].df, TAB_V2),
                                                                       None,
                                                                       np.full(len(LABELS), 0.5))["mAP"]) for kk in (k1, k2)}
        print(f"  {site} two-season(324+324): " + " ".join(f"{kk} {R['adaptation'][site]['two_season_648'][kk]['mAP_mean']:.3f}" for kk in (k1, k2))
              + f" | scratch {R['adaptation'][site]['scratch_648_season1_only'][k1]['mAP_mean']:.3f}/"
              f"{R['adaptation'][site]['scratch_648_season1_only'][k2]['mAP_mean']:.3f}", flush=True)

    tag = "_smoke" if a.smoke else ""
    with open(os.path.join(OUT_DIR, f"classification_results{tag}.json"), "w") as f:
        json.dump(R, f, indent=1, default=float)
    np.savez_compressed(os.path.join(OUT_DIR, f"classification_preds{tag}.npz"), **preds,
                        **{f"Y/{k}": v for k, v in Ytest.items()}, **{f"VC/{k}": v for k, v in VCtest.items()})
    print(f"\ndone in {(time.time() - t_start) / 60:.1f} min")


if __name__ == "__main__":
    main()
