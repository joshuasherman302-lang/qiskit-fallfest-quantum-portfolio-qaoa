"""Classical baselines. Each returns (basis-state index, number of cost evaluations).

These are deliberately simple and cheap: the point is to be honest about
what a laptop does to this problem before any quantum circuit runs.
"""
from __future__ import annotations

from itertools import combinations

import numpy as np
from scipy.optimize import minimize

from .hamiltonian import build_cost_hamiltonian
from .problem import PortfolioProblem


def _index(selected) -> int:
    return int(sum(1 << int(i) for i in selected))


def _x(problem: PortfolioProblem, selected) -> np.ndarray:
    x = np.zeros(problem.n)
    x[list(selected)] = 1.0
    return x


def brute_force(problem: PortfolioProblem):
    """Exact: evaluate every budget-feasible portfolio."""
    best, best_cost, evals = None, np.inf, 0
    for combo in combinations(range(problem.n), problem.budget):
        c = problem.cost(_x(problem, combo))
        evals += 1
        if c < best_cost:
            best, best_cost = combo, c
    return _index(best), evals


def top_return(problem: PortfolioProblem):
    """Naive: hold the `budget` assets with the highest expected return."""
    return _index(np.argsort(-problem.mu)[:problem.budget]), 0


def field_ranking(problem: PortfolioProblem):
    """Sort assets by their budget-gauge field and keep the top `budget`.

    This solves the additive part of the objective exactly and ignores the
    residual pair interactions. No cost evaluations, O(n^2) arithmetic.
    """
    ham = build_cost_hamiltonian(problem, 0.0, gauge=True, normalise=False)
    return _index(np.argsort(-ham.h)[:problem.budget]), 0


def swap_local_search(problem: PortfolioProblem, start_index: int):
    """Steepest-descent over single swaps (one asset out, one in)."""
    selected = set(np.flatnonzero(problem.selection(start_index)))
    current = problem.cost(_x(problem, selected))
    evals = 1
    improved = True
    while improved:
        improved = False
        best_move, best_cost = None, current
        for out in selected:
            for inn in set(range(problem.n)) - selected:
                c = problem.cost(_x(problem, (selected - {out}) | {inn}))
                evals += 1
                if c < best_cost - 1e-15:
                    best_move, best_cost = (out, inn), c
        if best_move is not None:
            selected = (selected - {best_move[0]}) | {best_move[1]}
            current, improved = best_cost, True
    return _index(selected), evals


def field_ranking_plus_search(problem: PortfolioProblem):
    start, _ = field_ranking(problem)
    return swap_local_search(problem, start)


def simulated_annealing(problem: PortfolioProblem, rng: np.random.Generator, steps: int = 400):
    """Metropolis over single swaps with a geometric temperature schedule."""
    selected = set(rng.choice(problem.n, problem.budget, replace=False).tolist())
    current = problem.cost(_x(problem, selected))
    best, best_cost, evals = set(selected), current, 1
    spread = max(np.abs(problem.mu).max() / problem.budget, 1e-6)
    t_hot, t_cold = spread, spread * 1e-3
    for step in range(steps):
        temp = t_hot * (t_cold / t_hot) ** (step / max(steps - 1, 1))
        out = rng.choice(sorted(selected))
        inn = rng.choice(sorted(set(range(problem.n)) - selected))
        cand = (selected - {out}) | {inn}
        c = problem.cost(_x(problem, cand))
        evals += 1
        if c < current or rng.random() < np.exp(-(c - current) / temp):
            selected, current = cand, c
            if c < best_cost:
                best, best_cost = set(cand), c
    return _index(best), evals


def markowitz_rounding(problem: PortfolioProblem):
    """Continuous long-only Markowitz weights, then keep the `budget` largest."""
    n = problem.n
    objective = lambda w: problem.risk_aversion * w @ problem.sigma @ w - problem.mu @ w
    result = minimize(objective, np.full(n, 1.0 / n), method="SLSQP",
                      bounds=[(0.0, 1.0)] * n,
                      constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1.0}])
    return _index(np.argsort(-result.x)[:problem.budget]), int(result.nfev)


def random_feasible(problem: PortfolioProblem, rng: np.random.Generator, shots: int):
    """Best of `shots` uniformly random budget-feasible portfolios."""
    best, best_cost = None, np.inf
    for _ in range(shots):
        combo = rng.choice(problem.n, problem.budget, replace=False)
        c = problem.cost(_x(problem, combo))
        if c < best_cost:
            best, best_cost = combo, c
    return _index(best), shots


def optimality_gap(problem: PortfolioProblem, index: int) -> float:
    """0 = optimal, 1 = worst feasible portfolio."""
    cost = problem.cost(problem.selection(index))
    gap = (cost - problem.best_cost) / (problem.worst_feasible_cost - problem.best_cost)
    # the two cost paths differ by float rounding only; anything larger is a real bug
    if gap < -1e-9:
        raise AssertionError(f"portfolio {index} beats the brute-force optimum by {-gap:.3g}")
    return float(max(gap, 0.0))
