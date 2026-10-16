"""
Figures for the weak-spot study and the ablation (reads classification_results.json / classification_preds.npz).

  fig9_weakspots.png          before/after for the four documented weak spots
  fig10_ablation.png          mAP heat-map: model x test set, incl. AdaBN
  fig11_label_efficiency.png  mAP vs number of labelled windows at the held-out sites, scratch / GB / two-season markers
  fig12_lofar_vessel_type.png narrowband comb separates cargo from tanker
  fig13_calibration.png       reliability diagrams before / after temperature scaling
  fig14_vessel_confusion.png  vessel-type confusion, v1 vs v2 inputs
"""

import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from simulation.ocean.dataset import LABELS, OUT_DIR, VESSEL_CLASSES, load_lofar, load_run  # noqa: E402
from simulation.ocean.lofar import F_LO, bin_hz  # noqa: E402

FIG = os.path.join(OUT_DIR, "figures")
R = json.load(open(os.path.join(OUT_DIR, "classification_results.json")))
Z = np.load(os.path.join(OUT_DIR, "classification_preds.npz"))
X = json.load(open(os.path.join(OUT_DIR, "extra_results.json")))
TESTS = list(R["tests"])
SHORT = {"A_unseen_weather": "A weather", "A_unseen_season": "A season", "B_season1": "B spring", "B_season2": "B summer",
         "E_season1": "E winter", "E_season2": "E autumn"}
C = {"v1": "#9aa5b1", "v2": "#1f77b4", "multi": "#2ca02c", "warn": "#d62728", "scr": "#ff7f0e"}
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 130})


def get(key, field, default=np.nan):
    v = R.get(key, {})
    return v.get(field, default) if isinstance(v, dict) else default


def fig9():
    fig, ax = plt.subplots(2, 2, figsize=(11, 7.5))
    models = [("CNN-Aonly-v1", "A-only, v1 inputs", C["v1"]), ("CNN-Aonly-v2", "A-only, + LOFAR/health", "#6baed6"),
              ("CNN-multi-v1", "multi-site, v1 inputs", "#a1d99b"), ("CNN-multi-v2", "multi-site, v2 inputs", C["v2"])]
    # (a) faults
    a = ax[0, 0]
    t = "A_unseen_weather"
    w = 0.2
    for i, (m, lab, col) in enumerate(models):
        ap = R[f"{m}/{t}"]["ap"]["fault_any"]
        a.bar(i * w * 5, ap, w * 4.2, color=col, label=lab)
        a.text(i * w * 5, ap + 0.01, f"{ap:.2f}", ha="center", fontsize=8)
    a.set_xticks([])
    a.set_ylim(0, 1.08)
    a.set_title("(a) Instrument-fault AP, unseen weather (v1 was 0.44)")
    a.legend(fontsize=7, loc="lower right")
    a.set_ylabel("average precision")
    # (b) cargo vs tanker, family acc
    a = ax[0, 1]
    xs = np.arange(len(TESTS))
    for i, (m, lab, col) in enumerate(models):
        v = [R[f"{m}/{t}"].get("cargo_vs_tanker_acc", np.nan) for t in TESTS]
        a.bar(xs + (i - 1.5) * 0.2, v, 0.19, color=col)
    a.axhline(0.5, color="k", lw=0.6, ls=":")
    a.set_xticks(xs)
    a.set_xticklabels([SHORT[t] for t in TESTS], rotation=20)
    a.set_ylim(0.4, 1.02)
    a.set_title("(b) Cargo vs. tanker accuracy (chance 0.5)")
    # (c) airgun AP on the FULL held-out runs, zero-shot
    a = ax[1, 0]
    runs = list(X["full_run_zero_shot"]["multi-v2"])
    xr = np.arange(len(runs))
    for i, (m, lab, col) in enumerate(models):
        tag = m.replace("CNN-", "")
        v = [X["full_run_zero_shot"][tag][r]["airgun_ap"] for r in runs]
        a.bar(xr + (i - 1.5) * 0.2, v, 0.19, color=col)
    a.set_xticks(xr)
    a.set_xticklabels([r.replace("_", chr(10), 1) for r in runs], fontsize=7)
    a.set_ylim(0, 1.05)
    a.set_title("(c) Seismic airgun AP, zero-shot, full runs (v1 at B: ~0)")
    a.text(3, 0.12, "B summer: all events at 10-13.5 dB,"+chr(10)+"the audibility limit", ha="center", fontsize=7, color=C["warn"])
    # (d) fine-tuning value depends on the pre-training
    a = ax[1, 1]
    for site, col in (("B", C["v2"]), ("E", C["multi"])):
        cur = X["ft_from_Aonly"][site]
        ns = sorted(int(k) for k in cur)
        a.plot(ns, [cur[str(n)]["season2"]["mAP"] for n in ns], marker="o", color=col, ls="--",
               label=f"site {site}: fine-tune from A-only model")
        a.axhline(R[f"CNN-multi-v2/{site}_season2"]["mAP"], color=col, lw=1.2, label=f"site {site}: multi-site model, zero labels")
    a.set_xlabel("labelled windows at the new site (other-season test)")
    a.set_ylabel("mAP, other season")
    a.set_title("(d) Other-season mAP: fine-tuning vs multi-site (v1: 0.809)")
    a.legend(fontsize=6.5, loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig9_weakspots.png"))
    plt.close(fig)


def fig10():
    rows = [("GB-v1-Aonly", "GB, A-only, v1"), ("GB-v1-multi", "GB, multi-site, v1"), ("GB-v2-multi", "GB, multi-site, v2"),
            ("CNN-Aonly-v1", "CNN, A-only, v1"), ("CNN-Aonly-v2", "CNN, A-only, v2"), ("CNN-multi-v1", "CNN, multi, v1"),
            ("CNN-multi-v2", "CNN, multi, v2"), ("CNN-multi-v2+AdaBN", "CNN multi v2 + AdaBN")]
    M = np.full((len(rows), len(TESTS)), np.nan)
    for i, (k, _) in enumerate(rows):
        for j, t in enumerate(TESTS):
            if f"{k}/{t}" in R:
                M[i, j] = R[f"{k}/{t}"]["mAP"]
    fig, a = plt.subplots(figsize=(7.6, 4.2))
    im = a.imshow(M, vmin=0.6, vmax=1.0, cmap="viridis", aspect="auto")
    a.set_xticks(range(len(TESTS)))
    a.set_xticklabels([SHORT[t] for t in TESTS])
    a.set_yticks(range(len(rows)))
    a.set_yticklabels([r[1] for r in rows])
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if not np.isnan(M[i, j]):
                a.text(j, i, f"{M[i, j]:.3f}", ha="center", va="center", fontsize=8, color="w" if M[i, j] < 0.85 else "k")
    a.set_title("mAP by model and test set (A: seen site, B/E: never seen in training)")
    fig.colorbar(im, ax=a, shrink=0.8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig10_ablation.png"))
    plt.close(fig)


def fig11():
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    for a, site in zip(ax, ("B", "E")):
        for key, col, ls in (("season1", C["v2"], "-"), ("season2", C["multi"], "--")):
            cur = R["adaptation"][site]["curve"][key]
            ns = sorted(int(k) for k in cur)
            mu = np.array([cur[str(n)]["mAP_mean"] for n in ns])
            sd = np.array([cur[str(n)]["mAP_std"] for n in ns])
            a.errorbar(ns, mu, sd, color=col, ls=ls, marker="o", capsize=2,
                       label="same season" if key == "season1" else "other season")
            sc = R["adaptation"][site]["scratch_648_season1_only"][f"{site}_{key}"]
            a.errorbar([648], [sc["mAP_mean"]], [sc["mAP_std"]], color=C["scr"], marker="s", capsize=2,
                       label="scratch, same 648 windows" if key == "season1" else None)
            gb = R["adaptation"][site]["gb_multi+648"][f"{site}_{key}"]
            a.scatter([648], [gb], marker="^", color="gray", zorder=5, label="GB + 648 target windows" if key == "season1" else None)
        ab = R[f"CNN-multi-v2+AdaBN/{site}_season1"]["mAP"]
        a.axhline(ab, color=C["v2"], lw=0.7, ls=":", label="AdaBN, 0 labels (same season)")
        a.set_title(f"held-out site {site}")
        a.set_xlabel("labelled windows (chronological, from deployment start)")
    ax[0].set_ylabel("mAP")
    ax[0].legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig11_label_efficiency.png"))
    plt.close(fig)


def fig12():
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.6))
    names = ["harbor_shelf_s2", "estuary_bay_s20", "strait_channel_s30", "offshore_shelf_s40"]
    f0 = {c: [] for c in VESSEL_CLASSES}
    lof_sum = {c: [] for c in ("cargo", "tanker")}
    for n in names:
        df, _ = load_run(n)
        lof = load_lofar(n).astype(np.float32)
        m = df["is_vessel"].values & (df["snr_vessel_db"].values >= 15)
        for c in VESSEL_CLASSES:
            f0[c] += df.loc[m & (df["vessel_class"].values == c), "f_comb_f0_hz"].tolist()
        for c in lof_sum:
            lof_sum[c].append(lof[m & (df["vessel_class"].values == c)])
    bins = np.linspace(0, 80, 81)
    for c in VESSEL_CLASSES:
        ax[0].hist(f0[c], bins=bins, alpha=0.55, label=f"{c} (n={len(f0[c])})")
    ax[0].set_xlabel("estimated comb spacing f0 (Hz)")
    ax[0].set_title("Blade-rate estimate per vessel class (SNR >= 15 dB)")
    ax[0].legend(fontsize=7)
    fr = F_LO + np.arange(lof_sum["cargo"][0].shape[1]) * bin_hz()
    df0, _ = load_run("harbor_shelf_s2")
    lof0 = load_lofar("harbor_shelf_s2").astype(np.float32)
    off = 0.0
    for c, col in (("cargo", C["v2"]), ("tanker", C["warn"])):
        m = (df0["vessel_class"].values == c) & df0["is_vessel"].values & (df0["f_comb_strength"].values > 0.25)
        i = np.nonzero(m)[0][np.argmax(df0["snr_vessel_db"].values[m])]
        ax[1].plot(fr, lof0[i] + off, color=col, lw=0.8, label=f"{c}: estimated spacing {df0['f_comb_f0_hz'].iloc[i]:.1f} Hz")
        f0 = df0["f_comb_f0_hz"].iloc[i]
        for h in range(1, 12):
            ax[1].axvline(f0 * h, color=col, lw=0.3, ls=":")
        off += 22
    ax[1].set_xlim(8, 90)
    ax[1].set_xlabel("frequency (Hz)")
    ax[1].set_ylabel("whitened level (dB, tanker offset +22)")
    ax[1].set_title("Single-window LOFAR with detected comb")
    ax[1].legend(fontsize=7)
    for m, col, lab in (("CNN-multi-v1", C["v1"], "v1 (mel only)"), ("CNN-multi-v2", C["v2"], "v2 (+ LOFAR/comb)")):
        v = [R[f"{m}/{t}"].get("cargo_vs_tanker_auc", np.nan) for t in TESTS]
        ax[2].plot(range(len(TESTS)), v, marker="o", color=col, label=lab)
    ax[2].axhline(0.5, color="k", lw=0.6, ls=":")
    ax[2].set_xticks(range(len(TESTS)))
    ax[2].set_xticklabels([SHORT[t] for t in TESTS], rotation=20)
    ax[2].set_ylabel("cargo-vs-tanker AUC")
    ax[2].set_title("Class separability, held-out data")
    ax[2].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig12_lofar_vessel_type.png"))
    plt.close(fig)


def fig13():
    fig, ax = plt.subplots(1, 4, figsize=(13, 3.3))
    labs = ["is_vessel", "storm", "is_airgun", "fault_any"]
    for a, lab in zip(ax, labs):
        j = LABELS.index(lab)
        for tag, col, name in (("CNN-multi-v2", C["warn"], "raw"), ("CNN-multi-v2-cal", C["v2"], "temperature-scaled")):
            p = np.concatenate([Z[f"{tag}/{t}"][:, j] for t in TESTS])
            y = np.concatenate([Z[f"Y/{t}"][:, j] for t in TESTS])
            edges = np.array([0, 0.02, 0.05, 0.1, 0.2, 0.35, 0.5, 0.65, 0.8, 0.9, 0.97, 1.0])
            idx = np.clip(np.digitize(p, edges) - 1, 0, len(edges) - 2)
            xs, ys = [], []
            for b in range(len(edges) - 1):
                m = idx == b
                if m.sum() >= 10:
                    xs.append(p[m].mean())
                    ys.append(y[m].mean())
            a.plot(xs, ys, marker="o", ms=3, color=col, label=name)
        a.plot([0, 1], [0, 1], "k:", lw=0.7)
        er = np.mean([R["calibration"]["ece_raw"][t][lab] for t in TESTS])
        es = np.mean([R["calibration"]["ece_scaled"][t][lab] for t in TESTS])
        a.set_title(f"{lab}  ECE {er:.3f} -> {es:.3f}", fontsize=8)
        a.set_xlabel("predicted probability")
    ax[0].set_ylabel("observed frequency")
    ax[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig13_calibration.png"))
    plt.close(fig)


def fig14():
    fig, ax = plt.subplots(1, 2, figsize=(9.5, 4))
    for a, tag, title in zip(ax, ("CNN-multi-v1", "CNN-multi-v2"), ("v1 inputs (mel + env)", "v2 inputs (+ LOFAR, comb)")):
        cm = np.zeros((5, 5))
        for t in TESTS:
            vc, p = Z[f"VC/{t}"], Z[f"{tag}-vc/{t}"]
            m = vc >= 0
            for true, pred in zip(vc[m], p[m].argmax(1)):
                cm[true, pred] += 1
        cmn = cm / np.maximum(cm.sum(1, keepdims=True), 1)
        a.imshow(cmn, vmin=0, vmax=1, cmap="Blues")
        for i in range(5):
            for j in range(5):
                a.text(j, i, f"{cmn[i, j]:.2f}", ha="center", va="center", fontsize=8, color="w" if cmn[i, j] > 0.5 else "k")
        a.set_xticks(range(5))
        a.set_yticks(range(5))
        a.set_xticklabels(VESSEL_CLASSES, rotation=30)
        a.set_yticklabels(VESSEL_CLASSES)
        acc = np.trace(cm) / cm.sum()
        a.set_title(f"{title}  acc {acc:.2f}  (pooled over all 6 test sets)", fontsize=8)
        a.set_xlabel("predicted")
    ax[0].set_ylabel("true")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig14_vessel_confusion.png"))
    plt.close(fig)


if __name__ == "__main__":
    for fn in (fig9, fig10, fig11, fig12, fig13, fig14):
        fn()
        print("wrote", fn.__name__)
