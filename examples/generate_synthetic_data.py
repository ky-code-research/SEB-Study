"""Generate fully artificial rows for structural smoke tests only.

The values are simulated from scratch and are not sampled, bootstrapped, or
perturbed from PROUD participants. They cannot reproduce paper estimates.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

DOMAINS = ["Gastrointestinal","Cardiac","Skin","Nervous","EyeEar","Genitourinary","Sleep","Sexual","OtherPsych"]
WEEKS = ["Week2","Week8","Week12","Week24"]

def generate(n_patients=120, seed=20261028):
    rng = np.random.default_rng(seed)
    rows = []
    for pid in range(1, n_patients + 1):
        age = rng.normal()
        sex = rng.choice(["Female", "Male"])
        education = rng.choice(["Junior high or below", "High school/vocational", "College or above"])
        first = rng.choice(["FirstEpisode", "Recurrent"])
        med = rng.choice(["SNRI", "SSRI"])
        bhamd, bsds = rng.normal(size=2)
        site = f"synthetic_site_{rng.integers(1, 7)}"
        for visit_order, week in enumerate(WEEKS, 1):
            x = rng.binomial(1, [0.30,0.28,0.18,0.25,0.13,0.12,0.32,0.16,0.15])
            ch = rng.normal()
            linear = -1.5 + 0.35*x[0] + 0.25*x[6] + 0.15*ch + 0.08*visit_order
            probs = np.array([1/(1+np.exp(linear)), .18, .12, .06])
            probs = probs / probs.sum()
            burden = int(rng.choice(4, p=probs))
            row = dict(participant_ID=f"SYN{pid:04d}", time=week, visit_order=visit_order,
                       burden_ord4=burden, fibser_burden=burden, age_z=age, sex=sex,
                       education3=education, education=education, first_episode=first,
                       baseline_HAMD17_z=bhamd, baseline_SDS_z=bsds,
                       baseline_med_class=med, concurrent_HAMD17_z=ch, site=site)
            row.update(dict(zip(DOMAINS, x)))
            rows.append(row)
    return pd.DataFrame(rows)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, default=Path("synthetic_analysis.csv"))
    ap.add_argument("--patients", type=int, default=120)
    args = ap.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    generate(args.patients).to_csv(args.output, index=False)
    print(f"Wrote {args.output}; synthetic patients={args.patients}")

if __name__ == "__main__":
    main()

