"""Antithesis tests for the Hamiltonian: try to break the cost identity."""
import numpy as np
import pytest

from qpo.hamiltonian import (build_cost_hamiltonian, build_defective_hamiltonian, verify)
from qpo.problem import PortfolioProblem


def random_problem(n, budget, seed):
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(n, n + 2))
    sigma = A @ A.T / (n + 2) * 0.05
    mu = rng.uniform(-0.1, 0.4, n)
    return PortfolioProblem(mu=mu, sigma=sigma, budget=budget, risk_aversion=rng.uniform(1, 6))


CASES = [(4, 2), (5, 2), (6, 3), (7, 2), (8, 4), (9, 6)]


@pytest.mark.parametrize("n,budget", CASES)
@pytest.mark.parametrize("lam", [0.0, 0.5, 2.0])
@pytest.mark.parametrize("normalise", [False, True])
def test_identity_holds_on_every_bitstring(n, budget, lam, normalise):
    problem = random_problem(n, budget, seed=n * 10 + budget)
    ham = build_cost_hamiltonian(problem, lam, gauge=False, normalise=normalise)
    report = verify(problem, ham)
    assert report["states_checked"] == 2 ** n
    assert report["max_abs_error"] < 1e-10
    assert report["operator_matches_coefficients"]


@pytest.mark.parametrize("n,budget", CASES)
def test_budget_gauge_is_exact_on_feasible_states(n, budget):
    problem = random_problem(n, budget, seed=n * 7 + budget)
    ham = build_cost_hamiltonian(problem, 0.0, gauge=True, normalise=True)
    report = verify(problem, ham)
    assert report["states_checked"] == problem.n_feasible
    assert report["max_abs_error"] < 1e-10
    assert report["feasible_argmin_preserved"]
    assert report["operator_matches_coefficients"]


def test_budget_gauge_is_not_valid_off_the_subspace():
    """The gauge is only an identity where the budget holds: confirm it does
    break elsewhere, so nobody uses it with a mixer that leaves the subspace."""
    problem = random_problem(6, 3, seed=1)
    ham = build_cost_hamiltonian(problem, 0.0, gauge=True, normalise=True)
    err = np.abs(ham.costs() - problem.cost_vector)
    assert err[~problem.feasible_mask].max() > 1e-6


def test_budget_gauge_shrinks_couplings():
    problem = random_problem(8, 4, seed=3)
    plain = build_cost_hamiltonian(problem, 0.0, gauge=False, normalise=False)
    gauged = build_cost_hamiltonian(problem, 0.0, gauge=True, normalise=False)
    assert np.abs(gauged.J).sum() < np.abs(plain.J).sum()


def test_defective_hamiltonian_is_caught():
    """The reconstruction of the reviewed defects must fail the check, with
    the dominant error equal to the dropped constant lam * B^2."""
    problem = random_problem(4, 2, seed=0)
    lam = 2.0
    report = verify(problem, build_defective_hamiltonian(problem, lam))
    assert report["max_abs_error"] > 1.0
    assert abs(report["max_abs_error"] - lam * problem.budget ** 2) < 0.5


def test_cost_matches_hand_calculation():
    mu = np.array([0.10, 0.20, 0.05])
    sigma = np.array([[0.04, 0.01, 0.00], [0.01, 0.09, 0.02], [0.00, 0.02, 0.01]])
    problem = PortfolioProblem(mu=mu, sigma=sigma, budget=2, risk_aversion=2.0)
    # assets 0 and 1, weights 1/2 each
    variance = (0.04 + 0.09 + 2 * 0.01) / 4
    expected = 2.0 * variance - (0.10 + 0.20) / 2
    assert problem.cost([1, 1, 0]) == pytest.approx(expected)
    assert problem.cost_vector[0b011] == pytest.approx(expected)
