"""
Stage 3 (supervised multi-label classification) + transfer/fine-tuning, on the physics-based ocean dataset.

Targets per window (multi-label): vessel, dolphin, whale, fish chorus, seismic airgun, storm, instrument fault;
plus a 5-way vessel-type head (cargo/tanker/ferry/fishing/outboard) trained on vessel windows.

Models
  GB   gradient boosting on calibrated tabular features (+ env sensors + clock), random-search tuned
  CNN  small convnet on the 64-band log-mel spectrogram, with env/clock vector fused in the head

Splits (no leakage: chronological blocks and whole unseen runs)
  train      site A runs s0,s1 + event-rich s4-s6, first 80%   val   same runs, last 20% (tuning / early stop / thresholds)
  test-A     site A s2          unseen weather, same site/season
  test-season site A s3         unseen season (summer vs spring)
  B zero-shot: site B (60 m deep, different species mix / traffic / noise) with the model trained on A only
  B fine-tune: B s10 first 15% = labelled adaptation set, next 10% = val, last 75% = test; B s11 = other season

  python simulation/scripts/ocean_classify.py
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
import torch.nn as nn
import torch.nn.functional as F
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, f1_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from simulation.ocean.dataset import (ACO_EXT, CTX, ENV_RAW, LABELS, OUT_DIR, VESSEL_CLASSES,  # noqa: E402
                                      load_run)

warnings.filterwarnings("ignore")
torch.set_num_threads(max(os.cpu_count() or 4, 4))
TAB = ACO_EXT + ENV_RAW + CTX
VEC = ENV_RAW + CTX


# ------------------------------------------------------------------ helpers
def split(df, mel, a, b):
    n = len(df)
    sl = slice(int(a * n), int(b * n))
    return df.iloc[sl], mel[sl]


def vclass_idx(df):
    m = {c: i for i, c in enumerate(VESSEL_CLASSES)}
    return np.array([m.get(c, -1) if isinstance(c, str) else -1 for c in df["vessel_class"]])


def metrics(Y, P, thr, vc_true=None, vc_prob=None):
    out = {"ap": {}, "f1": {}}
    for j, lab in enumerate(LABELS):
        y = Y[:, j]
        out["ap"][lab] = float(average_precision_score(y, P[:, j])) if 0 < y.sum() < len(y) else float("nan")
        out["f1"][lab] = float(f1_score(y, P[:, j] > thr[j], zero_division=0)) if y.sum() > 0 else float("nan")
    out["mAP"] = float(np.nanmean(list(out["ap"].values())))
    out["macroF1"] = float(np.nanmean(list(out["f1"].values())))
    if vc_true is not None:
        m = vc_true >= 0
        if m.sum() > 5:
            pred = vc_prob[m].argmax(1)
            out["vessel_type_acc"] = float((pred == vc_true[m]).mean())
            out["vessel_type_macroF1"] = float(f1_score(vc_true[m], pred, average="macro", zero_division=0))
    return out


def best_thresholds(Y, P):
    thr = np.full(Y.shape[1], 0.5)
    for j in range(Y.shape[1]):
        if Y[:, j].sum() == 0:
            continue
        cands = np.linspace(0.05, 0.95, 19)
        f = [f1_score(Y[:, j], P[:, j] > t, zero_division=0) for t in cands]
        thr[j] = cands[int(np.argmax(f))]
    return thr


# ------------------------------------------------------------------ GB model
def fit_gb(tr, va, trials=6, seed=0, sample_weight=None):
    rng = np.random.default_rng(seed)
    Xtr, Xva = tr[TAB].values, va[TAB].values if va is not None else None
    models, picked = [], []
    for j, lab in enumerate(LABELS):
        y, yv = tr[lab].values.astype(int), (va[lab].values.astype(int) if va is not None else None)
        if y.sum() < 3:
            models.append(None)
            picked.append(None)
            continue
        best, bm = -1, None
        for t in range(trials if va is not None else 1):
            kw = dict(learning_rate=float(rng.choice([0.04, 0.08, 0.15])), max_leaf_nodes=int(rng.choice([8, 15, 31])),
                      l2_regularization=float(rng.choice([0.0, 0.5, 2.0])), max_iter=int(rng.choice([150, 300])),
                      min_samples_leaf=int(rng.choice([10, 20])), random_state=seed)
            m = HistGradientBoostingClassifier(class_weight="balanced", **kw)
            m.fit(Xtr, y, sample_weight=sample_weight)
            sc = average_precision_score(yv, m.predict_proba(Xva)[:, 1]) if (va is not None and yv.sum() > 0) else 0.0
            if sc > best:
                best, bm, bkw = sc, m, kw
        models.append(bm)
        picked.append(bkw)
    return models, picked


def pred_gb(models, df):
    X = df[TAB].values
    return np.stack([m.predict_proba(X)[:, 1] if m is not None else np.zeros(len(df)) for m in models], axis=1)


# ------------------------------------------------------------------ CNN model
class Net(nn.Module):
    def __init__(self, n_vec, n_lab, n_vc, width=24):
        super().__init__()
        w = width
        chans = [1, w, 2 * w, 4 * w, 4 * w]
        self.blocks = nn.ModuleList()
        for i in range(4):
            self.blocks.append(nn.Sequential(
                nn.Conv2d(chans[i], chans[i + 1], 3, padding=1, bias=False), nn.BatchNorm2d(chans[i + 1]), nn.ReLU(),
                nn.Conv2d(chans[i + 1], chans[i + 1], 3, padding=1, bias=False), nn.BatchNorm2d(chans[i + 1]), nn.ReLU(),
                nn.MaxPool2d(2)))
        self.vec = nn.Sequential(nn.Linear(n_vec, 32), nn.ReLU())
        self.drop = nn.Dropout(0.3)
        d = 4 * w * 2 + 32
        self.head = nn.Linear(d, n_lab)
        self.vc_head = nn.Linear(d, n_vc)

    def forward(self, x, v):
        for b in self.blocks:
            x = b(x)
        z = torch.cat([x.mean((2, 3)), x.amax((2, 3)), self.vec(v)], dim=1)
        z = self.drop(z)
        return self.head(z), self.vc_head(z)


class Norm:
    def __init__(self, mel_ref, vec_ref):
        m = mel_ref.astype(np.float32)
        self.mu = m.mean(axis=(0, 2), keepdims=True)[None]
        self.sd = m.std(axis=(0, 2), keepdims=True)[None] + 1e-3
        self.vmed = np.median(vec_ref, axis=0)
        q75, q25 = np.percentile(vec_ref, [75, 25], axis=0)
        self.vsc = np.maximum((q75 - q25) / 1.349, vec_ref.std(0) * 0.25 + 1e-6)

    def mel(self, m):
        return torch.tensor((m.astype(np.float32)[:, None] - self.mu) / self.sd)

    def vec(self, v):
        return torch.tensor(np.clip((v - self.vmed) / self.vsc, -10, 10), dtype=torch.float32)


def to_tensors(norm, df, mel):
    return (norm.mel(mel), norm.vec(df[VEC].values), torch.tensor(df[LABELS].values.astype(np.float32)),
            torch.tensor(vclass_idx(df)))


def augment(x, v, gen):
    b = x.shape[0]
    x = x + (torch.rand(b, 1, 1, 1, generator=gen) - 0.5) * 0.6          # +-~3 dB gain / sensitivity drift
    for i in range(b):
        f0 = int(torch.randint(0, x.shape[2] - 8, (1,), generator=gen))
        x[i, :, f0:f0 + int(torch.randint(0, 8, (1,), generator=gen))] = 0.0
        t0 = int(torch.randint(0, x.shape[3] - 12, (1,), generator=gen))
        x[i, :, :, t0:t0 + int(torch.randint(0, 12, (1,), generator=gen))] = 0.0
    return x, v + 0.05 * torch.randn_like(v)


def predict(net, X, V, bs=256):
    net.eval()
    ps, vcs = [], []
    with torch.no_grad():
        for s in range(0, len(X), bs):
            lo, vc = net(X[s:s + bs], V[s:s + bs])
            ps.append(torch.sigmoid(lo))
            vcs.append(F.softmax(vc, 1))
    return torch.cat(ps).numpy(), torch.cat(vcs).numpy()


def mean_ap(Y, P):
    return float(np.nanmean([average_precision_score(Y[:, j], P[:, j]) if 0 < Y[:, j].sum() < len(Y) else np.nan
                             for j in range(Y.shape[1])]))


def train_net(net, train, val, epochs, lr, seed, trainable=None, patience=6, bs=96, log=None, wd=1e-3):
    X, V, Y, C = train
    gen = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    params = [p for n, p in net.named_parameters() if trainable is None or trainable(n)]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=wd)
    steps = epochs * int(np.ceil(len(X) / bs))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.2)
    pos = Y.sum(0).clamp(min=1)
    pw = torch.clamp(((len(Y) - pos) / pos).sqrt(), 1.0, 12.0)
    best, best_state, bad, hist = -1, None, 0, []
    for ep in range(epochs):
        net.train()
        perm = torch.randperm(len(X), generator=gen)
        tot = 0.0
        for s in range(0, len(X), bs):
            idx = perm[s:s + bs]
            if len(idx) < 4:
                continue
            xb, vb = augment(X[idx].clone(), V[idx], gen)
            lo, vc = net(xb, vb)
            loss = F.binary_cross_entropy_with_logits(lo, Y[idx], pos_weight=pw)
            m = C[idx] >= 0
            if m.any():
                loss = loss + 0.3 * F.cross_entropy(vc[m], C[idx][m])
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tot += float(loss) * len(idx)
        score = mean_ap(val[2].numpy(), predict(net, val[0], val[1])[0])
        hist.append((tot / len(X), score))
        if log:
            print(f"    ep{ep + 1:02d} loss {tot / len(X):.3f} val mAP {score:.3f}", flush=True)
        if score > best:
            best, best_state, bad = score, copy.deepcopy(net.state_dict()), 0
        else:
            bad += 1
            if bad >= patience:
                break
    net.load_state_dict(best_state)
    return net, hist, best


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=22)
    ap.add_argument("--ft-seeds", type=int, default=3)
    ap.add_argument("--width", type=int, default=24)
    a = ap.parse_args()
    t_start = time.time()

    R = {n: load_run(n) for n in ("harbor_shelf_s0", "harbor_shelf_s1", "harbor_shelf_s2", "harbor_shelf_s3",
                                  "open_coast_s10", "open_coast_s11",
                                  "rich_harbor_s4", "rich_harbor_s5", "rich_harbor_s6")}
    tr_parts, va_parts = [], []
    for n in ("harbor_shelf_s0", "harbor_shelf_s1", "rich_harbor_s4", "rich_harbor_s5", "rich_harbor_s6"):
        tr_parts.append(split(*R[n], 0.0, 0.8))
        va_parts.append(split(*R[n], 0.8, 1.0))
    A_tr = (pd.concat([p[0] for p in tr_parts]), np.concatenate([p[1] for p in tr_parts]))
    A_va = (pd.concat([p[0] for p in va_parts]), np.concatenate([p[1] for p in va_parts]))
    tests = {
        "A_test_unseen_weather": R["harbor_shelf_s2"],
        "A_test_other_season": R["harbor_shelf_s3"],
        "B_zero_shot": split(*R["open_coast_s10"], 0.25, 1.0),
        "B_other_season": R["open_coast_s11"],
    }
    B_ft = split(*R["open_coast_s10"], 0.0, 0.15)
    B_val = split(*R["open_coast_s10"], 0.15, 0.25)
    print("train", len(A_tr[0]), "val", len(A_va[0]), "B-ft", len(B_ft[0]), "B-val", len(B_val[0]))
    print("label prevalence (train): " + ", ".join(f"{l}={A_tr[0][l].mean():.3f}" for l in LABELS))

    results = {}
    preds = {}
    Ytest = {k: v[0][LABELS].values.astype(int) for k, v in tests.items()}

    # ---------------- GB
    print("\n== Gradient boosting (random-search tuned on val) ==")
    gbm, picked = fit_gb(A_tr[0], A_va[0])
    P_va = pred_gb(gbm, A_va[0])
    thr_gb = best_thresholds(A_va[0][LABELS].values.astype(int), P_va)
    for k, (d, m) in tests.items():
        P = pred_gb(gbm, d)
        results[f"GB/{k}"] = metrics(Ytest[k], P, thr_gb)
        preds[f"GB/{k}"] = P
        print(f"  {k:24s} mAP {results[f'GB/{k}']['mAP']:.3f}  macroF1 {results[f'GB/{k}']['macroF1']:.3f}")

    # ---------------- CNN pretrain on A
    print("\n== CNN pretrain on site A ==")
    norm = Norm(A_tr[1], A_tr[0][VEC].values)
    TR, VA = to_tensors(norm, *A_tr), to_tensors(norm, *A_va)
    net = Net(len(VEC), len(LABELS), len(VESSEL_CLASSES), a.width)
    print("  params:", sum(p.numel() for p in net.parameters()))
    net, hist, best = train_net(net, TR, VA, a.epochs, 2e-3, seed=0, log=True)
    torch.save({"state": net.state_dict(), "labels": LABELS, "width": a.width}, os.path.join(OUT_DIR, "cnn_pretrained.pt"))
    Pva, _ = predict(net, VA[0], VA[1])
    thr_cnn = best_thresholds(A_va[0][LABELS].values.astype(int), Pva)
    cnn_hist = hist
    for k, (d, m) in tests.items():
        T = to_tensors(norm, d, m)
        P, VC = predict(net, T[0], T[1])
        results[f"CNN/{k}"] = metrics(Ytest[k], P, thr_cnn, vclass_idx(d), VC)
        preds[f"CNN/{k}"] = P
        preds[f"CNNvc/{k}"] = VC
        r = results[f"CNN/{k}"]
        print(f"  {k:24s} mAP {r['mAP']:.3f}  macroF1 {r['macroF1']:.3f}  vessel-type acc {r.get('vessel_type_acc', float('nan')):.3f}")

    # ---------------- fine-tuning at site B
    print("\n== Fine-tuning on a small labelled B set ==")
    FT, FV = to_tensors(norm, *B_ft), to_tensors(norm, *B_val)
    Yft_val = B_val[0][LABELS].values.astype(int)
    ft_results = {"adapt_windows": len(B_ft[0]), "configs": [], "final": {}}
    stages = {"head_only": lambda n: n.startswith(("head", "vc_head", "vec")),
              "last_block+head": lambda n: n.startswith(("blocks.3", "head", "vc_head", "vec")),
              "all_layers": None}
    sel = []
    pretrained = {"state": copy.deepcopy(net.state_dict())}
    for stage, fn in stages.items():
        for lr in (3e-4, 1e-3):
            n2 = Net(len(VEC), len(LABELS), len(VESSEL_CLASSES), a.width)
            n2.load_state_dict(pretrained["state"])
            n2, _, sc = train_net(n2, FT, FV, 14, lr, seed=0, trainable=fn)
            ft_results["configs"].append({"stage": stage, "lr": lr, "val_mAP": sc})
            print(f"  FT {stage:16s} lr={lr:g} val mAP {sc:.3f}")
            sel.append((sc, stage, lr))
    sc0, st0, lr0 = max(sel)
    print(f"  selected: {st0}, lr={lr0:g}")
    ft_results["selected"] = {"stage": st0, "lr": lr0}

    ft_runs, scratch_runs = [], []
    for sd in range(a.ft_seeds):
        n2 = Net(len(VEC), len(LABELS), len(VESSEL_CLASSES), a.width)
        n2.load_state_dict(pretrained["state"])
        n2, _, _ = train_net(n2, FT, FV, 14, lr0, seed=100 + sd, trainable=stages[st0])
        Pv, _ = predict(n2, FV[0], FV[1])
        thr = best_thresholds(Yft_val, Pv)
        ns = Net(len(VEC), len(LABELS), len(VESSEL_CLASSES), a.width)
        ns, _, _ = train_net(ns, FT, FV, 30, 2e-3, seed=100 + sd)
        Pv2, _ = predict(ns, FV[0], FV[1])
        thr2 = best_thresholds(Yft_val, Pv2)
        for tag, model, th, store in (("finetuned", n2, thr, ft_runs), ("scratch", ns, thr2, scratch_runs)):
            row = {}
            for k in ("B_zero_shot", "B_other_season"):
                d, m = tests[k]
                T = to_tensors(norm, d, m)
                P, VC = predict(model, T[0], T[1])
                row[k] = metrics(Ytest[k], P, th, vclass_idx(d), VC)
                if sd == 0:
                    preds[f"CNN-{tag}/{k}"] = P
                    preds[f"CNN-{tag}vc/{k}"] = VC
            store.append(row)

    def agg(runs, k, key):
        v = [r[k][key] for r in runs]
        return float(np.mean(v)), float(np.std(v))

    for k in ("B_zero_shot", "B_other_season"):
        zs = results[f"CNN/{k}"]
        ft_results["final"][k] = {
            "zero_shot_mAP": zs["mAP"], "zero_shot_macroF1": zs["macroF1"],
            "finetuned_mAP": agg(ft_runs, k, "mAP"), "finetuned_macroF1": agg(ft_runs, k, "macroF1"),
            "scratch_mAP": agg(scratch_runs, k, "mAP"), "scratch_macroF1": agg(scratch_runs, k, "macroF1"),
            "finetuned_per_label_ap": {l: float(np.nanmean([r[k]["ap"][l] for r in ft_runs])) for l in LABELS},
            "zero_shot_per_label_ap": zs["ap"],
            "scratch_per_label_ap": {l: float(np.nanmean([r[k]["ap"][l] for r in scratch_runs])) for l in LABELS},
        }
        f = ft_results["final"][k]
        print(f"  {k}: zero-shot mAP {f['zero_shot_mAP']:.3f} | fine-tuned {f['finetuned_mAP'][0]:.3f}±{f['finetuned_mAP'][1]:.3f} "
              f"| scratch(same {len(B_ft[0])} windows) {f['scratch_mAP'][0]:.3f}±{f['scratch_mAP'][1]:.3f}")

    # GB trained on B-small only / A + B-small
    both = pd.concat([A_tr[0], B_ft[0]])
    w = np.concatenate([np.ones(len(A_tr[0])), np.full(len(B_ft[0]), 4.0)])
    gb_b, _ = fit_gb(pd.concat([A_tr[0], B_ft[0]]), None, sample_weight=w)
    for k in ("B_zero_shot", "B_other_season"):
        P = pred_gb(gb_b, tests[k][0])
        ft_results["final"][k]["gb_A+B_small_mAP"] = metrics(Ytest[k], P, thr_gb)["mAP"]

    results["finetune"] = ft_results
    results["gb_hyperparams"] = picked
    results["cnn_pretrain_history"] = cnn_hist
    results["label_prevalence"] = {k: {l: float(v[0][l].mean()) for l in LABELS} for k, v in tests.items()}
    results["thresholds_cnn"] = thr_cnn.tolist()
    with open(os.path.join(OUT_DIR, "classification_results.json"), "w") as f:
        json.dump(results, f, indent=1, default=float)
    np.savez_compressed(os.path.join(OUT_DIR, "classification_preds.npz"), **preds,
                        **{f"Y/{k}": v for k, v in Ytest.items()},
                        **{f"VC/{k}": vclass_idx(v[0]) for k, v in tests.items()})
    print(f"\ndone in {(time.time() - t_start) / 60:.1f} min")


if __name__ == "__main__":
    main()
