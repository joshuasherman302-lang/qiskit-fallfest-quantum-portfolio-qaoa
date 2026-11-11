"""E1. Is the Hamiltonian the cost function?

Brute-force comparison on the 4-asset toy: the corrected build, and a
reconstruction of the two defects found in review.
"""
from common import save

import numpy as np

from qpo.data import annualised_stats, load_prices
from qpo.hamiltonian import build_cost_hamiltonian, build_defective_hamiltonian, verify
from qpo.problem import PortfolioProblem

LAMBDA = 2.0
TICKERS = ("AAPL", "JPM", "XOM", "JNJ")


def main():
    prices = load_prices()
    mu, sigma, names = annualised_stats(prices, TICKERS, "2024-01-01", "2025-12-31")
    problem = PortfolioProblem(mu, sigma, budget=2, risk_aversion=3.0, tickers=names)

    correct = build_cost_hamiltonian(problem, LAMBDA, normalise=False)
    defective = build_defective_hamiltonian(problem, LAMBDA)
    gauged = build_cost_hamiltonian(problem, 0.0, gauge=True, normalise=False)

    truth = problem.penalised_cost_vector(LAMBDA)
    rows = []
    for index in np.flatnonzero(problem.feasible_mask):
        rows.append({
            "portfolio": "+".join(problem.names(int(index))),
            "bitstring": problem.bitstring(int(index)),
            "true_cost": float(truth[index]),
            "corrected_H": float(correct.costs()[index]),
            "defective_H": float(defective.costs()[index]),
        })
    rows.sort(key=lambda r: r["true_cost"])

    feasible = problem.feasible_costs
    spread = float(feasible.max() - feasible.min())
    penalty_gap = float(truth[~problem.feasible_mask].min() - feasible.min())
    payload = {
        "tickers": names, "budget": 2, "risk_aversion": 3.0, "lambda": LAMBDA,
        "window": "2024-01-01..2025-12-31",
        "verify_corrected": verify(problem, correct),
        "verify_defective": verify(problem, defective),
        "verify_gauged": verify(problem, gauged),
        "feasible_table": rows,
        "feasible_cost_spread": spread,
        "penalty_gap": penalty_gap,
        "penalty_gap_over_spread": penalty_gap / spread,
        "max_abs_coupling_penalised": float(np.abs(correct.J).max()),
        "max_abs_coupling_gauged": float(np.abs(gauged.J).max()),
        "max_abs_field_gauged": float(np.abs(gauged.h).max()),
        "defective_ranks_feasible_correctly": bool(
            np.array_equal(np.argsort(defective.costs()[problem.feasible_mask]),
                           np.argsort(feasible))),
    }
    save("e01_hamiltonian_check", payload)

    print(f"\n{'portfolio':12s} {'true':>9s} {'corrected':>10s} {'defective':>10s}")
    for r in rows:
        print(f"{r['portfolio']:12s} {r['true_cost']:9.4f} {r['corrected_H']:10.4f} {r['defective_H']:10.4f}")
    print(f"\ncorrected  max |error| over 16 states: {payload['verify_corrected']['max_abs_error']:.2e}")
    print(f"defective  max |error| over 16 states: {payload['verify_defective']['max_abs_error']:.3f}"
          f"   (lambda * B^2 = {LAMBDA * 4:.1f})")
    print(f"penalty gap / feasible spread: {payload['penalty_gap_over_spread']:.1f}x")
    print(f"largest coupling: penalised {payload['max_abs_coupling_penalised']:.4f} -> "
          f"budget gauge {payload['max_abs_coupling_gauged']:.5f}")


if __name__ == "__main__":
    main()
