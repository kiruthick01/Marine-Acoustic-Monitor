"""
Render figures for the physics-based ocean simulation (data figures + model-result figures).

  python simulation/scripts/ocean_render.py            # everything available
  python simulation/scripts/ocean_render.py --only data
"""

import argparse
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from simulation.ocean import physics as ph  # noqa: E402
from simulation.ocean.dataset import LABELS, OUT_DIR, VESSEL_CLASSES, load_run  # noqa: E402
from simulation.ocean.weather import SITE_A, sound_speed_profile  # noqa: E402

FIG = os.path.join(OUT_DIR, "figures")
os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 130, "axes.grid": True, "grid.alpha": 0.25,
                     "axes.spines.top": False, "axes.spines.right": False, "font.size": 9})
C = {"wind": "#2a6fdb", "rain": "#7a3fc4", "temp": "#d9480f", "sal": "#0b7285", "turb": "#8d6e43", "ph": "#2b8a3e",
     "storm": "#d6336c", "vessel": "#e8590c", "dolphin": "#1c7ed6", "whale": "#5f3dc4", "fish": "#2f9e44",
     "airgun": "#c92a2a", "fault": "#495057"}


def save(fig, name):
    p = os.path.join(FIG, name)
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print("wrote", p)


def shade(ax, t, mask, color, alpha=0.18):
    m = np.asarray(mask, dtype=bool)
    if not m.any():
        return
    edges = np.flatnonzero(np.diff(np.concatenate([[0], m.astype(int), [0]])))
    for s, e in zip(edges[::2], edges[1::2]):
        ax.axvspan(t[s], t[min(e, len(t) - 1)], color=color, alpha=alpha, lw=0)


# ---------------------------------------------------------------- physics
def fig_physics():
    fig, ax = plt.subplots(2, 3, figsize=(15, 8))
    z = np.linspace(0, 20, 100)
    for lab, (sst, td, st, col) in {"summer (stratified)": (22, 12, 0.9, "#d9480f"), "spring": (16, 11, 0.4, "#2f9e44"),
                                    "winter (mixed)": (11, 10.5, 0.05, "#1c7ed6")}.items():
        t, c = sound_speed_profile(z, sst, td, st, 35.0, 20.0)
        ax[0, 0].plot(c, z, color=col, label=lab)
    ax[0, 0].invert_yaxis()
    ax[0, 0].set(xlabel="sound speed (m/s)", ylabel="depth (m)", title="Sound-speed profile (Mackenzie): seasonal thermocline")
    ax[0, 0].legend()

    f = np.logspace(1, 4.2, 200)
    for phv, ls in ((7.9, "--"), (8.05, "-"), (8.2, ":")):
        ax[0, 1].loglog(f, ph.absorption_db_per_km(f, 15, 35, phv, 10), ls, color="#0b7285", label=f"pH {phv}")
    for T, col in ((5, "#1c7ed6"), (25, "#d9480f")):
        ax[0, 1].loglog(f, ph.absorption_db_per_km(f, T, 35, 8.05, 10), color=col, alpha=0.7, label=f"{T} °C (pH 8.05)")
    ax[0, 1].set(xlabel="frequency (Hz)", ylabel="absorption (dB/km)", title="Seawater absorption (Ainslie-McColm): pH and T matter")
    ax[0, 1].legend()

    ff = np.logspace(1, np.log10(11000), 300)
    for w, col in ((1, "#74c0fc"), (5, "#339af0"), (10, "#1971c2"), (15, "#0b3d91")):
        ax[0, 2].semilogx(ff, ph.wenz_psd_db(ff, w, shipping=0.5), color=col, label=f"wind {w} m/s")
    ax[0, 2].semilogx(ff, ph.wenz_psd_db(ff, 3, shipping=0.5, rain_mm_h=8), color=C["rain"], ls="--", label="3 m/s + rain 8 mm/h")
    ax[0, 2].semilogx(ff, ph.wenz_psd_db(ff, 3, shipping=1.0), color=C["vessel"], ls="--", label="3 m/s + heavy shipping")
    ax[0, 2].semilogx(ff, ph.wenz_psd_db(ff, 3, shipping=0.3, flow_ms=0.5), color="#868e96", ls="--", label="3 m/s + 0.5 m/s tidal flow")
    ax[0, 2].set(xlabel="frequency (Hz)", ylabel="PSD (dB re 1 µPa²/Hz)", title="Ambient noise (Wenz/Coates + rain + flow)")
    ax[0, 2].legend(fontsize=7)

    r = np.logspace(1.5, 4.5, 100)
    for fr, col in ((50, "#c92a2a"), (100, "#e8590c"), (1000, "#2f9e44"), (8000, "#1c7ed6")):
        ax[1, 0].semilogx(r, ph.transmission_loss_db(fr, r, 15, 35, 8.05, 10, 20), color=col, label=f"{fr} Hz")
    ax[1, 0].invert_yaxis()
    ax[1, 0].set(xlabel="range (m)", ylabel="transmission loss (dB)",
                 title=f"TL, 20 m shelf: cutoff {ph.waveguide_cutoff_hz(20):.0f} Hz, practical spreading")
    ax[1, 0].legend()

    for kn, L, col, name in ((13, 250, "#364fc7", "cargo 13 kn"), (20, 60, "#e8590c", "ferry 20 kn"), (22, 8, "#2b8a3e", "outboard 22 kn")):
        sl = ph.ross_source_level_db(kn, L)
        ax[1, 1].semilogx(ff, ph.vessel_broadband_psd_db(ff, sl + {250: 12, 60: 6, 8: 8}[L]), color=col, label=f"{name}")
    ax[1, 1].set(xlabel="frequency (Hz)", ylabel="source PSD (dB re 1 µPa²/Hz @ 1 m)", title="Vessel cavitation continuum (Ross-shaped)")
    ax[1, 1].legend()

    wind = np.linspace(0.5, 20, 60)
    ax[1, 2].plot(wind, ph.sea_state_hs(wind), color=C["wind"], label="fully developed Hs (PM)")
    ax[1, 2].plot(wind, np.minimum(ph.sea_state_hs(wind) * SITE_A.fetch_factor, 6), color=C["temp"], label="fetch-limited (site A)")
    ax[1, 2].set(xlabel="wind speed (m/s)", ylabel="significant wave height (m)", title="Wind → waves")
    ax[1, 2].legend()
    fig.suptitle("Physics layer of the ocean simulator", y=1.01, fontsize=13)
    save(fig, "fig1_physics.png")


# ---------------------------------------------------------------- environment
def fig_environment(df, name):
    t = df["elapsed_minutes"].values / 1440.0
    fig, ax = plt.subplots(8, 1, figsize=(14, 17), sharex=True)
    a = ax[0]; a.plot(t, df.wind_ms, color=C["wind"], lw=0.8, label="wind 10 m"); a.plot(t, df.gust_ms, color=C["wind"], alpha=0.3, lw=0.6, label="gust")
    a.set_ylabel("wind (m/s)"); a2 = a.twinx(); a2.bar(t, df.rain_mm_h, width=0.007, color=C["rain"], alpha=0.7); a2.set_ylabel("rain (mm/h)", color=C["rain"]); a2.grid(False)
    a.legend(loc="upper left", fontsize=7)
    a = ax[1]; a.plot(t, df.pressure_hpa, color="#495057", lw=0.8); a.set_ylabel("pressure (hPa)")
    a2 = a.twinx(); a2.plot(t, df.hs_m, color=C["wind"], lw=0.9); a2.set_ylabel("Hs waves (m)", color=C["wind"]); a2.grid(False)
    a = ax[2]; a.plot(t, df.tide_m, color="#1864ab", lw=0.8); a.set_ylabel("tide (m)")
    a2 = a.twinx(); a2.plot(t, df.current_ms, color="#868e96", lw=0.6); a2.set_ylabel("current (m/s)"); a2.grid(False)
    a = ax[3]; a.plot(t, df.air_temp_c, color="#adb5bd", lw=0.7, label="air"); a.plot(t, df.sst_c, color=C["temp"], lw=1, label="SST")
    a.plot(t, df.temperature_c_true, color="#862e9c", lw=1, label="at hydrophone"); a.plot(t, df.temperature_c, color="k", lw=0.4, alpha=0.6, label="sensor")
    a.set_ylabel("temperature (°C)"); a.legend(ncol=4, fontsize=7, loc="upper left")
    a2 = a.twinx(); a2.fill_between(t, 0, df.stratification, color="#fcc419", alpha=0.25); a2.set_ylim(0, 3); a2.set_yticks([]); a2.grid(False)
    a = ax[4]; a.plot(t, df.salinity_psu_true, color=C["sal"], lw=1); a.plot(t, df.salinity_psu, color="k", lw=0.3, alpha=0.5); a.set_ylabel("salinity (PSU)")
    a2 = a.twinx(); a2.plot(t, df.runoff_index, color="#e67700", lw=0.9); a2.set_ylabel("runoff index", color="#e67700"); a2.grid(False)
    a = ax[5]; a.plot(t, df.turbidity_ntu_true, color=C["turb"], lw=1, label="truth"); a.plot(t, df.turbidity_ntu, color="k", lw=0.3, alpha=0.5, label="sensor (spikes, drift, stuck)")
    a.set_ylabel("turbidity (NTU)"); a.legend(fontsize=7, loc="upper left")
    a = ax[6]; a.plot(t, df.ph_true, color=C["ph"], lw=1); a.plot(t, df.ph, color="k", lw=0.3, alpha=0.5); a.set_ylabel("pH")
    a2 = a.twinx(); a2.plot(t, df.dissolved_o2_mg_l, color="#1098ad", lw=0.8); a2.set_ylabel("O₂ (mg/L)", color="#1098ad"); a2.grid(False)
    a = ax[7]; a.plot(t, df.sound_speed_ms, color="#5c940d", lw=0.9); a.set_ylabel("sound speed (m/s)"); a.set_xlabel("day of run")
    for a in ax:
        shade(a, t, df.storm, C["storm"], 0.10)
        shade(a, t, df.fault_env, C["fault"], 0.25)
    ax[0].set_title(f"Coupled atmosphere-ocean state and sensor readings — {name}   (pink = storm/runoff windows, grey = stuck-sensor fault)")
    fig.tight_layout()
    save(fig, f"fig2_environment_{name}.png")


# ---------------------------------------------------------------- soundscape
def fig_soundscape(df, mel, name):
    t = df["elapsed_minutes"].values / 1440.0
    freqs = np.linspace(20, 11000, mel.shape[1])  # mel axis shown by index
    ltsa = mel.astype(np.float32).mean(axis=2).T
    fig = plt.figure(figsize=(15, 9))
    gs = fig.add_gridspec(4, 1, height_ratios=[3.2, 1.4, 1.6, 1.0], hspace=0.08)
    a0 = fig.add_subplot(gs[0])
    im = a0.imshow(ltsa, aspect="auto", origin="lower", extent=[t[0], t[-1], 0, mel.shape[1]], cmap="magma",
                   vmin=np.percentile(ltsa, 2), vmax=np.percentile(ltsa, 99.5))
    a0.set_ylabel("mel band (low→high freq)"); a0.set_title(f"Long-term spectral average, 5 s every 10 min — {name}"); a0.grid(False)
    plt.colorbar(im, ax=a0, pad=0.01, label="dB re full scale")
    a1 = fig.add_subplot(gs[1], sharex=a0)
    for col, lab, c in (("f_band_50_100_db", "50-100 Hz (vessels)", C["vessel"]), ("f_band_500_1000_db", "0.5-1 kHz", C["fish"]),
                        ("f_band_5000_11000_db", "5-11 kHz (wind, rain, shrimp)", C["wind"])):
        a1.plot(t, df[col], lw=0.6, color=c, label=lab)
    a1.set_ylabel("band level (dBFS)"); a1.legend(ncol=3, fontsize=7, loc="lower right")
    a2 = fig.add_subplot(gs[2], sharex=a0)
    rows = [("vessel", df.is_vessel), ("airgun", df.is_airgun), ("storm", df.storm), ("fault", df.fault_any),
            ("dolphin", df.is_dolphin), ("whale", df.is_whale), ("fish", df.is_fish)]
    for i, (lab, m) in enumerate(rows):
        a2.vlines(t[m.values], i - 0.4, i + 0.4, color=C[lab], lw=1.2)
    a2.set_yticks(range(len(rows))); a2.set_yticklabels([r[0] for r in rows]); a2.set_ylabel("ground-truth events"); a2.grid(False)
    a3 = fig.add_subplot(gs[3], sharex=a0)
    a3.plot(t, df.wind_ms, color=C["wind"], lw=0.8); a3.set_ylabel("wind (m/s)"); a3.set_xlabel("day of run")
    for a in (a0, a1, a2):
        plt.setp(a.get_xticklabels(), visible=False)
    save(fig, f"fig3_soundscape_{name}.png")


def fig_examples(df, mel, name):
    picks = [("calm ambient", (~df.anomaly) & (~df.is_dolphin) & (~df.is_whale) & (~df.is_fish)),
             ("storm / rain", df.storm & ~df.is_vessel & (df.rain_mm_h > 2)),
             ("windy sea", df.storm & ~df.is_vessel & (df.wind_ms > 12)),
             ("cargo vessel", df.is_vessel & (df.vessel_class == "cargo")),
             ("tanker", df.is_vessel & (df.vessel_class == "tanker")),
             ("ferry", df.is_vessel & (df.vessel_class == "ferry")),
             ("fishing boat", df.is_vessel & (df.vessel_class == "fishing")),
             ("outboard", df.is_vessel & (df.vessel_class == "outboard")),
             ("dolphin whistles+clicks", df.is_dolphin), ("baleen whale song", df.is_whale), ("fish chorus", df.is_fish),
             ("seismic airgun", df.is_airgun), ("50/60 Hz hum fault", df.fault_audio == "hum"),
             ("cable pops fault", df.fault_audio == "pops")]
    fig, ax = plt.subplots(3, 5, figsize=(17, 9))
    vmin, vmax = np.percentile(mel[:200].astype(np.float32), 1), np.percentile(mel.astype(np.float32), 99.9)
    for a, (title, m) in zip(ax.ravel(), picks):
        idx = np.flatnonzero(m.values)
        a.grid(False)
        if len(idx) == 0:
            a.set_title(title + " (none)", fontsize=9); a.axis("off"); continue
        sel = idx[np.argmax(df["snr_vessel_db"].values[idx])] if "vessel" in title or title in ("cargo vessel", "tanker", "ferry", "fishing boat", "outboard") else idx[len(idx) // 2]
        a.imshow(mel[sel].astype(np.float32), aspect="auto", origin="lower", cmap="magma", vmin=vmin, vmax=vmax,
                 extent=[0, 5, 20, 11000])
        a.set_yscale("symlog", linthresh=500)
        a.set_title(f"{title}", fontsize=9)
    for a in ax.ravel()[len(picks):]:
        a.axis("off")
    for a in ax[:, 0]:
        a.set_ylabel("Hz")
    for a in ax[-1]:
        a.set_xlabel("s")
    fig.suptitle(f"Example 5-second windows (log-mel), {name}", fontsize=13)
    fig.tight_layout()
    save(fig, f"fig4_examples_{name}.png")


def fig_coupling(dfs):
    df = pd.concat(dfs)
    fig, ax = plt.subplots(2, 3, figsize=(15, 8))
    quiet = ~df.is_vessel & ~df.is_airgun
    s = ax[0, 0].scatter(df.wind_ms[quiet], df.f_band_5000_11000_db[quiet], c=df.rain_mm_h[quiet], s=4, cmap="viridis_r", vmax=6)
    ax[0, 0].set(xlabel="wind (m/s)", ylabel="5-11 kHz band level (dBFS)", title="High-frequency noise vs. wind and rain"); plt.colorbar(s, ax=ax[0, 0], label="rain mm/h")
    ax[0, 1].scatter(df.flow_ms[quiet], df.f_band_20_50_db[quiet], s=4, c=df.wind_ms[quiet], cmap="Blues")
    ax[0, 1].set(xlabel="tidal flow (m/s)", ylabel="20-50 Hz band level (dBFS)", title="Mooring flow noise vs. tidal current")
    ax[0, 2].scatter(df.runoff_index, df.salinity_psu_true, s=4, c=df.rain_mm_h, cmap="Purples")
    ax[0, 2].set(xlabel="runoff index", ylabel="salinity (PSU)", title="Rain → runoff → freshening")
    ax[1, 0].scatter(df.hs_m, df.turbidity_ntu_true, s=4, color=C["turb"], alpha=0.5)
    ax[1, 0].set(xlabel="significant wave height (m)", ylabel="turbidity (NTU)", title="Wave resuspension + runoff → turbidity")
    d = df.groupby(df.hour.round()).agg(ph=("ph_true", "mean"), fish=("is_fish", "mean"), dol=("is_dolphin", "mean"), sh=("bio_shrimp_rate", "mean"))
    ax[1, 1].plot(d.index, d.ph, color=C["ph"], marker="o", ms=3); ax[1, 1].set(xlabel="hour of day", ylabel="mean pH", title="Diel pH cycle (photosynthesis) + biology by hour")
    a2 = ax[1, 1].twinx(); a2.plot(d.index, d.fish, color=C["fish"], label="fish chorus"); a2.plot(d.index, d.dol * 2, color=C["dolphin"], label="dolphin ×2"); a2.legend(fontsize=7, loc="upper left"); a2.grid(False)
    ax[1, 2].scatter(df.temperature_c_true, df.bio_shrimp_rate, s=4, color="#e8590c", alpha=0.4)
    ax[1, 2].set(xlabel="water temperature (°C)", ylabel="snapping-shrimp rate (snaps/s)", title="Shrimp chorus vs. temperature")
    fig.suptitle("Environment ↔ acoustics ↔ biology couplings (site A, all runs)", fontsize=13)
    fig.tight_layout()
    save(fig, "fig5_couplings.png")


def run_data_figs():
    runs = {n: load_run(n) for n in ("harbor_shelf_s0", "harbor_shelf_s1", "harbor_shelf_s2", "open_coast_s10")
            if os.path.exists(os.path.join(OUT_DIR, n))}
    fig_physics()
    for n, (df, mel) in runs.items():
        if n in ("harbor_shelf_s0", "open_coast_s10"):
            fig_environment(df, n)
            fig_soundscape(df, mel, n)
            fig_examples(df, mel, n)
    fig_coupling([d for n, (d, m) in runs.items() if n.startswith("harbor")])


# ---------------------------------------------------------------- model results
def fig_detection():
    p = os.path.join(OUT_DIR, "detection_results.json")
    if not os.path.exists(p):
        return
    R = json.load(open(p))
    names = R["methods"]
    keys = list(R["results"])
    fig, ax = plt.subplots(1, 3, figsize=(17, 5))
    ap = [np.mean([v["auprc"] for v in R["results"][k].values()]) for k in keys]
    f1 = [np.mean([v["f1"] for v in R["results"][k].values()]) for k in keys]
    apsd = [np.std([v["auprc"] for v in R["results"][k].values()]) for k in keys]
    f1sd = [np.std([v["f1"] for v in R["results"][k].values()]) for k in keys]
    x = np.arange(len(keys)); w = 0.38
    ax[0].bar(x - w / 2, ap, w, yerr=apsd, color="#364fc7", label="AUPRC", capsize=3)
    ax[0].bar(x + w / 2, f1, w, yerr=f1sd, color="#e8590c", label="F1 @ calibration-quantile threshold", capsize=3)
    sup = np.mean([v["auprc"] for v in R["supervised_ceiling"].values()])
    ax[0].axhline(sup, color="k", ls="--", lw=0.8); ax[0].text(len(keys) - 0.5, sup + 0.01, "supervised (labelled) ceiling", ha="right", fontsize=7)
    ax[0].set_xticks(x); ax[0].set_xticklabels([f"{k}\n{names[k].split(' ', 1)[1][:26]}" for k in keys], fontsize=7)
    ax[0].set_title("Unsupervised detectors on unseen runs (mean ± sd)"); ax[0].legend(fontsize=7); ax[0].set_ylim(0, 1)
    types = ["vessel", "storm", "airgun", "fault"]
    wd = 0.12
    for i, k in enumerate(keys):
        vals = [np.nanmean([v[f"recall_{t}"] for v in R["results"][k].values()]) for t in types]
        ax[1].bar(np.arange(4) + (i - len(keys) / 2) * wd, vals, wd, label=k)
    ax[1].set_xticks(range(4)); ax[1].set_xticklabels(types); ax[1].set_ylim(0, 1.05); ax[1].legend(fontsize=7, ncol=3)
    ax[1].set_title("Recall by anomaly type (storm/env. dilution diagnosed in docs)")
    runs = list(R["results"][R["best_unsupervised"]])
    ax[2].bar(np.arange(len(runs)) - 0.2, [R["results"]["M0"][r]["auprc"] for r in runs], 0.2, label="M0 baseline", color="#adb5bd")
    ax[2].bar(np.arange(len(runs)), [R["results"][R["best_unsupervised"]][r]["auprc"] for r in runs], 0.2, label=f"{R['best_unsupervised']} best", color="#364fc7")
    ax[2].bar(np.arange(len(runs)) + 0.2, [R["supervised_ceiling"][r]["auprc"] for r in runs], 0.2, label="supervised", color="#2f9e44")
    ax[2].set_xticks(range(len(runs))); ax[2].set_xticklabels([r.replace("_", "\n") for r in runs], fontsize=7); ax[2].legend(fontsize=7)
    ax[2].set_title("AUPRC by run (spring/summer, site A/B)")
    fig.tight_layout()
    save(fig, "fig6_detection_summary.png")

    sp = os.path.join(OUT_DIR, "detection_scores.json")
    if os.path.exists(sp):
        S = json.load(open(sp))
        run = "harbor_shelf_s0" if "harbor_shelf_s0" in S else list(S)[0]
        df, _ = load_run(run)
        cal = R["cal_windows"]
        s = np.array(S[run]["score"])
        t = df["elapsed_minutes"].values[cal:] / 1440.0
        fig, ax = plt.subplots(2, 1, figsize=(15, 6), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
        ax[0].plot(t, s, lw=0.6, color="#364fc7")
        y = df["anomaly"].values[cal:]
        shade(ax[0], t, y, C["storm"], 0.15)
        pred = np.array(S[run]["pred"]).astype(bool)
        ax[0].scatter(t[pred], s[pred], s=5, color="#c92a2a", label="flagged", zorder=3)
        ax[0].set_ylabel("anomaly score"); ax[0].legend(); ax[0].set_title(f"Best unsupervised detector ({R['best_unsupervised']}) on unseen run {run}; shaded = true anomaly windows")
        rows = [("vessel", "is_vessel"), ("airgun", "is_airgun"), ("storm", "storm"), ("fault", "fault_any")]
        for i, (lab, col) in enumerate(rows):
            m = df[col].values[cal:]
            ax[1].vlines(t[m], i - 0.4, i + 0.4, color=C[lab], lw=1)
        ax[1].vlines(t[pred], len(rows) - 0.4, len(rows) + 0.4, color="#c92a2a", lw=1)
        ax[1].set_yticks(range(len(rows) + 1)); ax[1].set_yticklabels([r[0] for r in rows] + ["FLAGGED"]); ax[1].set_xlabel("day of run"); ax[1].grid(False)
        fig.tight_layout()
        save(fig, "fig7_detection_timeline.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["data", "models"], default=None)
    a = ap.parse_args()
    if a.only in (None, "data"):
        run_data_figs()
    if a.only in (None, "models"):
        fig_detection()
    