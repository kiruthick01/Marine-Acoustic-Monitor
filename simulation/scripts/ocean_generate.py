"""
Generate the physics-based ocean datasets used by the detection / classification / render scripts.

  python simulation/scripts/ocean_generate.py             # full corpus (~1 h on 16 cores)
  python simulation/scripts/ocean_generate.py --quick     # 6-day smoke set
  python simulation/scripts/ocean_generate.py --only open_coast_s10 rich_estuary_s7

Corpus design (domain randomisation + held-out sites):
  A harbor_shelf     20 m   train (s0, s1), test: unseen weather (s2) and other season (s3)
  C estuary_bay      32 m   train only, river-driven, busy traffic
  D strait_channel   48 m   train only, cold, strong tides
  B open_coast       60 m   HELD OUT, never seen in pre-training (spring s10, summer s11)
  E offshore_shelf  110 m   HELD OUT, quiet, whale-rich (winter s40, autumn s41)
  rich_*  event-rich training-only runs (forced seismic campaigns, many instrument faults), 15 days
Every non-rich run contains a seismic survey campaign so airgun detection is evaluable at every site.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from simulation.ocean.scenario import run_scenario  # noqa: E402
from simulation.ocean.weather import SITE_A, SITE_B, SITE_C, SITE_D, SITE_E  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "..", "output", "ocean")

# name, site, seed, start day-of-year, days factor, airgun_prob, fault_prob
RUNS = [
    ("harbor_shelf_s0", SITE_A, 0, 90, 1.0, 1.0, 0.6),
    ("harbor_shelf_s1", SITE_A, 1, 90, 1.0, 1.0, 0.6),
    ("harbor_shelf_s2", SITE_A, 2, 90, 1.0, 1.0, 0.6),
    ("harbor_shelf_s3", SITE_A, 3, 200, 1.0, 1.0, 0.6),
    ("estuary_bay_s20", SITE_C, 20, 280, 1.0, 1.0, 0.6),
    ("estuary_bay_s21", SITE_C, 21, 200, 1.0, 1.0, 0.6),
    ("strait_channel_s30", SITE_D, 30, 20, 1.0, 1.0, 0.6),
    ("strait_channel_s31", SITE_D, 31, 150, 1.0, 1.0, 0.6),
    ("open_coast_s10", SITE_B, 10, 90, 1.0, 1.0, 0.6),
    ("open_coast_s11", SITE_B, 11, 200, 1.0, 1.0, 0.6),
    ("offshore_shelf_s40", SITE_E, 40, 20, 1.0, 1.0, 0.6),
    ("offshore_shelf_s41", SITE_E, 41, 280, 1.0, 1.0, 0.6),
    ("rich_harbor_s4", SITE_A, 4, 90, 0.5, 1.0, 0.95),
    ("rich_harbor_s5", SITE_A, 5, 200, 0.5, 1.0, 0.95),
    ("rich_harbor_s6", SITE_A, 6, 150, 0.5, 1.0, 0.95),
    ("rich_estuary_s7", SITE_C, 7, 280, 0.5, 1.0, 0.95),
    ("rich_strait_s8", SITE_D, 8, 20, 0.5, 1.0, 0.95),
]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=30)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--only", nargs="*", default=None, help="run names to (re)generate")
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--n-jobs", type=int, default=-1)
    a = ap.parse_args()
    days = 6 if a.quick else a.days
    for name, site, seed, doy, factor, ag, fp in RUNS:
        if a.only and name not in a.only:
            continue
        if a.skip_existing and os.path.exists(os.path.join(OUT, name, "lofar.npy")):
            continue
        run_scenario(site=site, n_days=max(int(days * factor), 2), seed=seed, start_doy=doy, airgun_prob=ag,
                     fault_prob=fp, out_dir=os.path.join(OUT, name), n_jobs=a.n_jobs)
