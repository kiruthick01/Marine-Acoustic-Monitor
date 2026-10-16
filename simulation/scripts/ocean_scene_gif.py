"""
Animated scene: a 5-minute continuous slice (one 5 s window per frame) with a vessel passing the
hydrophone while a dolphin pod swims past, rendered by the same physics as the datasets.
Left: top-down map. Right: scrolling log-mel spectrogram. Writes figures/scene_vessel_dolphins.gif.

  python simulation/scripts/ocean_scene_gif.py
"""

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from simulation.ocean import anthro  # noqa: E402
from simulation.ocean.scenario import synthesize_window  # noqa: E402
from simulation.ocean.weather import SITE_A, simulate_environment  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "..", "output", "ocean", "figures")
N = 60  # frames = windows, 5 s each


def build(seed=3):
    rng = np.random.default_rng(seed)
    env = simulate_environment(200, SITE_A, seed=2).iloc[60].to_dict()
    env["wind_ms"], env["rain_mm_h"] = 4.0, 0.0
    envs = [dict(env) for _ in range(N)]
    zeros = lambda v: np.full(N, v)  # noqa: E731
    dp = np.zeros(N, bool); dp[18:46] = True
    drng = np.full(N, np.nan); drng[18:46] = np.linspace(520, 140, 28)
    bio = dict(shrimp_rate=zeros(60.0), fish_level=zeros(0.0), fish_f0=zeros(95.0), fish_callers=np.zeros(N, int),
               dolphin_present=dp, dolphin_range=drng, dolphin_n=np.where(dp, 4, 0),
               whale_present=np.zeros(N, bool), whale_range=zeros(np.nan))
    p = anthro.VesselPassage(cls="ferry", t_cpa_s=150.0, cpa_m=320.0, speed_ms=10.0, length_m=60.0, shaft_hz=8.0,
                             blades=4, source_level_db=ross_sl(), tonal_offsets=tuple(rng.uniform(0, 6.28, 10)))
    sh = dict(site=SITE_A, seed=seed, interval_min=5.0 / 60.0, passages=[p], airgun=[], faults=[], env=envs, bio=bio,
              sensors=[{} for _ in range(N)], vessel_candidates=[[0]] * N)
    return sh, p, drng


def ross_sl():
    from simulation.ocean.physics import ross_source_level_db
    return ross_source_level_db(19.4, 60.0) + 6.0


def main():
    sh, p, drng = build()
    mels, info = [], []
    rng = np.random.default_rng(5)
    bearing = rng.uniform(0, 2 * np.pi)
    for i in range(N):
        row, mel = synthesize_window(i, sh)
        mels.append(mel.astype(np.float32))
        r, vr, x = anthro.vessel_geometry(p, i * 5.0 + 2.5, SITE_A.hydrophone_depth_m)
        bearing += rng.normal(0, 0.15)
        info.append((float(x), float(r), float(vr), row["snr_vessel_db"], row["snr_dolphin_db"], bearing))
    vmin, vmax = np.percentile(np.stack(mels), 2), np.percentile(np.stack(mels), 99.7)

    fig = plt.figure(figsize=(13, 6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.5])
    am = fig.add_subplot(gs[0]); asp = fig.add_subplot(gs[1])
    am.set_xlim(-800, 800); am.set_ylim(-700, 700); am.set_aspect("equal"); am.set_facecolor("#e7f5ff")
    am.set_title("Top view (hydrophone at origin)")
    for rr in (200, 400, 600):
        am.add_patch(plt.Circle((0, 0), rr, fill=False, ls=":", color="#868e96"))
    am.plot([-800, 800], [p.cpa_m, p.cpa_m], color="#adb5bd", lw=0.8)
    am.plot(0, 0, "k^", ms=11, label="hydrophone")
    vessel, = am.plot([], [], "s", color="#e8590c", ms=11, label="ferry")
    pod = am.scatter([], [], c="#1c7ed6", s=30, label="dolphin pod")
    am.legend(loc="lower left", fontsize=8)
    txt = am.text(-780, 640, "", fontsize=9, va="top")
    im = asp.imshow(np.zeros((64, 108 * 6)), aspect="auto", origin="lower", cmap="magma", vmin=vmin, vmax=vmax, extent=[-30, 0, 20, 11000])
    asp.set_yscale("symlog", linthresh=500); asp.set_xlabel("seconds ago"); asp.set_ylabel("Hz"); asp.grid(False)
    asp.set_title("Hydrophone log-mel spectrogram (last 30 s)")
    dr = np.random.default_rng(1).normal(0, 25, (N, 4, 2))

    def update(i):
        x, r, vr, sv, sd, b = info[i]
        vessel.set_data([x], [p.cpa_m])
        if not np.isnan(drng[i]):
            c = np.array([np.cos(b), np.sin(b)]) * drng[i] * 0.8
            pod.set_offsets(c + dr[i])
        else:
            pod.set_offsets(np.empty((0, 2)))
        lo = max(0, i - 5)
        stack = np.concatenate(mels[lo:i + 1], axis=1)
        pad = 108 * 6 - stack.shape[1]
        stack = np.pad(stack, ((0, 0), (pad, 0)), constant_values=vmin)
        im.set_data(stack)
        doppler = 8.0 * 4 / (1 + vr / 1510.0)
        txt.set_text(f"t = {i * 5:3d} s\nferry range {r:5.0f} m, radial speed {vr:+.1f} m/s\n"
                     f"blade-rate tone {doppler:.2f} Hz (nominal 32.00)\nvessel SNR {sv:5.1f} dB   dolphin SNR {sd:5.1f} dB")
        return vessel, pod, im, txt

    ani = FuncAnimation(fig, update, frames=N, interval=250, blit=False)
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, "scene_vessel_dolphins.gif")
    ani.save(path, writer=PillowWriter(fps=4))
    print("wrote", path)


if __name__ == "__main__":
    main()
