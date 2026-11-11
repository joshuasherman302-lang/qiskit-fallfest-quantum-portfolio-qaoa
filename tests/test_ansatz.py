"""Antithesis tests for the circuits: parameter order, budget conservation,
and agreement between the fast simulator and Qiskit on the real circuits."""
import numpy as np
import pytest
from qiskit.quantum_info import Statevector

from qpo.ansatz import (bind_vector, build_qaoa_circuit, fourier_schedule, initial_selection)
from qpo.fastsim import FastQAOA
from qpo.hamiltonian import build_cost_hamiltonian
from tests.test_hamiltonian import random_problem


def qiskit_probs(circuit, betas, gammas):
    bound = circuit.assign_parameters(bind_vector(betas, gammas))
    return Statevector(bound).probabilities()


def test_fourier_schedule_matches_formula():
    amps = np.array([0.7, -0.2, 0.4, 0.1])     # u1, u2, v1, v2
    p = 8
    betas, gammas = fourier_schedule(amps, p)
    assert betas.shape == gammas.shape == (p,)
    for layer in range(1, p + 1):
        g = sum(amps[k - 1] * np.sin((k - 0.5) * (layer - 0.5) * np.pi / p) for k in (1, 2))
        b = sum(amps[2 + k - 1] * np.cos((k - 0.5) * (layer - 0.5) * np.pi / p) for k in (1, 2))
        assert gammas[layer - 1] == pytest.approx(g)
        assert betas[layer - 1] == pytest.approx(b)


@pytest.mark.parametrize("p", [1, 3, 12])
@pytest.mark.parametrize("mixer", ["x", "xy"])
def test_parameter_order_is_beta_then_gamma(p, mixer):
    problem = random_problem(4, 2, seed=5)
    ham = build_cost_hamiltonian(problem, 0.5 if mixer == "x" else 0.0, gauge=(mixer == "xy"))
    circuit = build_qaoa_circuit(ham, p, mixer, budget=2)
    names = [param.name for param in circuit.parameters]
    assert names == [f"β[{i}]" for i in range(p)] + [f"γ[{i}]" for i in range(p)]


def test_swapping_beta_and_gamma_changes_the_state():
    """If the order were ever swapped silently, this is what would differ."""
    problem = random_problem(5, 2, seed=2)
    ham = build_cost_hamiltonian(problem, 0.5)
    circuit = build_qaoa_circuit(ham, 2, "x")
    betas, gammas = np.array([0.3, 0.1]), np.array([0.9, 1.4])
    right = Statevector(circuit.assign_parameters(bind_vector(betas, gammas))).probabilities()
    wrong = Statevector(circuit.assign_parameters(bind_vector(gammas, betas))).probabilities()
    assert np.abs(right - wrong).max() > 1e-3


@pytest.mark.parametrize("n,budget", [(4, 2), (5, 2), (6, 3), (7, 3)])
@pytest.mark.parametrize("p", [1, 3])
def test_fast_simulator_matches_qiskit_x_mixer(n, budget, p):
    rng = np.random.default_rng(n + p)
    problem = random_problem(n, budget, seed=n)
    ham = build_cost_hamiltonian(problem, 0.5, gauge=False, normalise=True)
    betas, gammas = rng.uniform(-1, 1, p), rng.uniform(-1, 1, p)
    fast = FastQAOA(ham, "x").probabilities(betas, gammas)
    ref = qiskit_probs(build_qaoa_circuit(ham, p, "x"), betas, gammas)
    assert np.allclose(fast, ref, atol=1e-10)


@pytest.mark.parametrize("n,budget", [(4, 2), (5, 2), (6, 3), (7, 3)])
@pytest.mark.parametrize("p", [1, 3])
def test_fast_simulator_matches_qiskit_xy_mixer(n, budget, p):
    rng = np.random.default_rng(10 * n + p)
    problem = random_problem(n, budget, seed=n)
    ham = build_cost_hamiltonian(problem, 0.0, gauge=True, normalise=True)
    betas, gammas = rng.uniform(-1, 1, p), rng.uniform(-1, 1, p)
    fast = FastQAOA(ham, "xy", budget).probabilities(betas, gammas)
    ref = qiskit_probs(build_qaoa_circuit(ham, p, "xy", budget), betas, gammas)
    assert np.allclose(fast, ref, atol=1e-10)


@pytest.mark.parametrize("n,budget", [(4, 2), (6, 2), (7, 4), (8, 4)])
def test_xy_mixer_never_leaves_the_budget(n, budget):
    rng = np.random.default_rng(n)
    problem = random_problem(n, budget, seed=n + 1)
    ham = build_cost_hamiltonian(problem, 0.0, gauge=True)
    probs = qiskit_probs(build_qaoa_circuit(ham, 4, "xy", budget),
                         rng.uniform(-2, 2, 4), rng.uniform(-2, 2, 4))
    assert probs[~problem.feasible_mask].sum() < 1e-12
    assert probs[problem.feasible_mask].sum() == pytest.approx(1.0)


def test_xy_mixer_can_reach_every_feasible_state():
    """The chain mixer must connect the whole feasible subspace."""
    rng = np.random.default_rng(0)
    problem = random_problem(6, 3, seed=9)
    ham = build_cost_hamiltonian(problem, 0.0, gauge=True)
    probs = FastQAOA(ham, "xy", 3).probabilities(rng.uniform(0.3, 1.2, 6), rng.uniform(0.3, 1.2, 6))
    assert (probs[problem.feasible_mask] > 1e-9).all()


def test_initial_selection_has_the_right_weight():
    for n in range(3, 13):
        for budget in range(1, n):
            sel = initial_selection(n, budget)
            assert len(sel) == budget and max(sel) < n
