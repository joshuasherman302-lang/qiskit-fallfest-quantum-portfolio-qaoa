"""Shared helpers for the experiment scripts."""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

import numpy as np                                   # noqa: E402
from scipy.optimize import minimize                  # noqa: E402

from qpo.ansatz import fourier_schedule              # noqa: E402
from qpo.metrics import bootstrap_ci                 # noqa: E402

RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

# The regime used throughout: risk-dominated selection, where pair
# interactions matter most (see e05_classical.py for why).
RISK_AVERSION = 30.0


def save(name: str, payload: dict) -> Path:
    path = RESULTS / f"{name}.json"
    path.write_text(json.dumps(payload, indent=2, default=float))
    print(f"wrote {path.relative_to(ROOT)}")
    return path


def ci(values) -> dict:
    """Median with a 95% percentile-bootstrap interval, plus n."""
    med, lo, hi = bootstrap_ci(values)
    return {"median": med, "lo": lo, "hi": hi, "n": int(len(values))}


def ceiling_angles(sim, p: int, starts: int, rng, fourier: bool = False):
    """Best angles found by multi-start L-BFGS on the EXACT energy.

    This is what the ansatz can do when tuning is not the bottleneck. It is a
    lower bound on the true optimum of the ansatz (finite number of starts).
    """
    best = None
    for _ in range(starts):
        if fourier:
            fun = lambda a: sim.energy(*fourier_schedule(a, p))
            x0 = rng.uniform(-1.0, 1.0, 4)
        else:
            fun = lambda x: sim.energy(x[:p], x[p:])
            x0 = rng.uniform(-1.0, 1.0, 2 * p)
        result = minimize(fun, x0, method="L-BFGS-B")
        if best is None or result.fun < best.fun:
            best = result
    if fourier:
        return fourier_schedule(best.x, p), best.x
    return (best.x[:p], best.x[p:]), best.x
