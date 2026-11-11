"""Scoring a QAOA output distribution against the brute-force answer.

Every number here has a "what would guessing score?" twin, because at toy
sizes a sampler that knows nothing still finds the optimum quickly.
"""
from __future__ import annotations

import numpy as np

from .problem import PortfolioProblem


def summarise(problem: PortfolioProblem, probs: np.ndarray) -> dict:
    """Metrics for a probability vector over all 2^n basis states."""
    probs = np.asarray(probs, dtype=float)
    feas = problem.feasible_mask
    p_feasible = float(probs[feas].sum())
    p_best = float(probs[problem.best_index])
    best, worst = problem.best_cost, problem.worst_feasible_cost
    span = worst - best
    out = {
        "p_feasible": p_feasible,
        "p_best": p_best,
        "p_feasible_uniform": problem.n_feasible / probs.size,   # uniform over all bitstrings
        "p_best_given_feasible_uniform": 1.0 / problem.n_feasible,
        "ratio_uniform": float((worst - problem.feasible_costs.mean()) / span),
    }
    if p_feasible > 0:
        cond = probs[feas] / p_feasible
        expected = float(cond @ problem.feasible_costs)
        out["p_best_given_feasible"] = p_best / p_feasible
        out["expected_cost_feasible"] = expected
        # 1.0 = always the optimum, 0.0 = always the worst feasible portfolio
        out["ratio"] = float((worst - expected) / span)
        # how many times likelier than a uniform feasible guess to hit the optimum
        out["enhancement"] = out["p_best_given_feasible"] * problem.n_feasible
    else:
        out.update(p_best_given_feasible=0.0, expected_cost_feasible=float("nan"),
                   ratio=float("nan"), enhancement=0.0)
    return out


def counts_to_probs(counts: dict, n: int) -> np.ndarray:
    """Qiskit counts ({'0101': 12, ...}, asset 0 right-most) -> probability vector."""
    probs = np.zeros(2 ** n)
    total = sum(counts.values())
    for bitstring, c in counts.items():
        probs[int(bitstring.replace(" ", ""), 2)] += c / total
    return probs


def bootstrap_ci(values, stat=np.median, n_boot: int = 4000, alpha: float = 0.05, seed: int = 0):
    """Percentile bootstrap confidence interval for a statistic of a 1-D sample."""
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, values.size, size=(n_boot, values.size))
    boots = stat(values[idx], axis=1)
    return float(stat(values)), float(np.quantile(boots, alpha / 2)), float(np.quantile(boots, 1 - alpha / 2))
