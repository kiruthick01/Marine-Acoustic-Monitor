"""
Generate the physics-based ocean datasets used by ocean_train.py / ocean_render.py.

  python simulation/scripts/ocean_generate.py            # full set (~20 min)
  python simulation/scripts/ocean_generate.py --quick    # 6-day smoke set
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from simulation.ocean.scenario import run_scenario  # noqa: E402
from simulation.ocean.weather import SITE_A, SITE_B  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "..", "output", "ocean")

# (site, seed, start day-of-year)  -- A: pretraining/test site, B: held-out domain for fine-tuning
RUNS = [(SITE_A, 0, 90), (SITE_A, 1, 90), (SITE_A, 2, 90), (SITE_A, 3, 200),
        (SITE_B, 10, 90), (SITE_B, 11, 200)]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=30)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--only-rich", action="store_true")
    ap.add_argument("--n-jobs", type=int, default=-1)
    a = ap.parse_args()
    days = 6 if a.quick else a.days
    if a.only_rich:
        RUNS.clear()
    # event-rich TRAINING-ONLY runs (forced seismic campaigns + many instrument faults), so the supervised
    # classifier sees rare classes; excluded from detection/test evaluation by name prefix "rich_"
    for seed, doy in ((4, 90), (5, 200), (6, 150)):
        run_scenario(site=SITE_A, n_days=days // 2, seed=seed, start_doy=doy, airgun_prob=1.0, fault_prob=0.95,
                     out_dir=os.path.join(OUT, f"rich_harbor_s{seed}"), n_jobs=a.n_jobs)
    for site, seed, doy in RUNS:
        run_scenario(site=site, n_days=days, seed=seed, start_doy=doy,
                     out_dir=os.path.join(OUT, f"{site.name}_s{seed}"), n_jobs=a.n_jobs)
