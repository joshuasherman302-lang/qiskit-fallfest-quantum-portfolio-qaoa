"""E2. What can each circuit do when tuning is NOT the bottleneck?

For every variant, angles are optimised on the exact energy with multi-start
L-BFGS. If a variant still scores at guessing level here, no optimiser
(SPSA, Bayesian, PPO) can rescue it.

Variants
  x_penalty : textbook QAOA, sum-X mixer, budget enforced by a penalty.
              The penalty weight is set to the feasible cost spread, an
              oracle choice that favours this baseline.
  xy_spread : XY mixer from one basis state (a random feasible portfolio,
              because the asset order of each instance is random).
  xy_pairs  : XY mixer from one entangled pair per two assets.
  xy_dicke  : XY mixer from the equal superposition of all feasible portfolios.
"""
from common import RISK_AVERSION, ceiling_angles, ci, save

import numpy as np

from qpo.data import load_prices
from qpo.fastsim import FastQAOA
from qpo.hamiltonian import build_cost_hamiltonian
from qpo.instances import make_pool
from qpo.metrics import summarise

SIZES = (4, 6, 8)
DEPTHS = (1, 2, 4)
N_INSTANCES = 20
STARTS = 8


def simulator(problem, variant):
    if variant == "x_penalty":
        spread = problem.worst_feasible_cost - problem.best_cost
        return FastQAOA(build_cost_hamiltonian(problem, lam=spread), "x")
    init = variant.split("_")[1]
    return FastQAOA(build_cost_hamiltonian(problem, 0.0, gauge=True), "xy", problem.budget, init)


def main():
    prices = load_prices()
    rng = np.random.default_rng(2026)
    out = {"risk_aversion": RISK_AVERSION, "n_instances": N_INSTANCES, "starts": STARTS, "cells": []}
    for n in SIZES:
        pool = make_pool(prices, "test", N_INSTANCES, n, n // 2, RISK_AVERSION, seed=100 + n)
        for variant in ("x_penalty", "xy_spread", "xy_pairs", "xy_dicke"):
            for p in DEPTHS:
                rows = []
                for problem in pool:
                    sim = simulator(problem, variant)
                    (betas, gammas), _ = ceiling_angles(sim, p, STARTS, rng)
                    rows.append(summarise(problem, sim.probabilities(betas, gammas)))
                cell = {
                    "n": n, "variant": variant, "p": p,
                    "p_feasible": ci([r["p_feasible"] for r in rows]),
                    "ratio": ci([r["ratio"] for r in rows]),
                    "enhancement": ci([r["enhancement"] for r in rows]),
                    "ratio_uniform": float(np.median([r["ratio_uniform"] for r in rows])),
                    "p_feasible_uniform": rows[0]["p_feasible_uniform"],
                }
                out["cells"].append(cell)
                print(f"n={n} {variant:10s} p={p}: feasible {cell['p_feasible']['median']:.2f}  "
                      f"ratio {cell['ratio']['median']:.3f} [{cell['ratio']['lo']:.3f}, {cell['ratio']['hi']:.3f}]  "
                      f"(guess {cell['ratio_uniform']:.3f})  "
                      f"P(best|feasible) = {cell['enhancement']['median']:.1f}x guess")
    save("e02_ansatz_ceiling", out)


if __name__ == "__main__":
    main()
