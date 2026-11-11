"""Dicke-state circuit and classical baselines."""
from math import comb

import numpy as np
import pytest
from qiskit.quantum_info import Statevector

from qpo import classical
from qpo.ansatz import bind_vector, build_qaoa_circuit
from qpo.dicke import dicke_circuit
from qpo.fastsim import FastQAOA
from qpo.hamiltonian import build_cost_hamiltonian
from tests.test_hamiltonian import random_problem


@pytest.mark.parametrize("n,k", [(2, 1), (3, 1), (3, 2), (4, 2), (5, 2), (6, 3), (7, 5), (8, 4)])
def test_dicke_circuit_prepares_the_exact_state(n, k):
    amps = Statevector(dicke_circuit(n, k)).data
    weight = np.array([bin(z).count("1") for z in range(2 ** n)])
    target = (weight == k) / np.sqrt(comb(n, k))
    phase = amps[np.argmax(np.abs(amps))] / abs(amps[np.argmax(np.abs(amps))])
    assert np.allclose(amps / phase, target, atol=1e-10)


@pytest.mark.parametrize("n,budget", [(4, 2), (6, 3), (7, 3)])
def test_fast_simulator_matches_qiskit_with_dicke_start(n, budget):
    rng = np.random.default_rng(n)
    problem = random_problem(n, budget, seed=n + 3)
    ham = build_cost_hamiltonian(problem, 0.0, gauge=True, normalise=True)
    betas, gammas = rng.uniform(-1, 1, 2), rng.uniform(-1, 1, 2)
    circuit = build_qaoa_circuit(ham, 2, "xy", budget, init="dicke")
    ref = Statevector(circuit.assign_parameters(bind_vector(betas, gammas))).probabilities()
    fast = FastQAOA(ham, "xy", budget, init="dicke").probabilities(betas, gammas)
    assert np.allclose(fast, ref, atol=1e-10)
    assert ref[~problem.feasible_mask].sum() < 1e-12


@pytest.mark.parametrize("seed", range(5))
def test_brute_force_agrees_with_the_cost_table(seed):
    problem = random_problem(7, 3, seed)
    index, evals = classical.brute_force(problem)
    assert index == problem.best_index
    assert evals == problem.n_feasible


@pytest.mark.parametrize("seed", range(5))
def test_every_baseline_returns_a_feasible_portfolio(seed):
    problem = random_problem(8, 4, seed)
    rng = np.random.default_rng(seed)
    for index, _ in (classical.top_return(problem), classical.field_ranking(problem),
                     classical.field_ranking_plus_search(problem),
                     classical.simulated_annealing(problem, rng, 200),
                     classical.markowitz_rounding(problem),
                     classical.random_feasible(problem, rng, 20)):
        assert problem.feasible_mask[index]
        assert 0.0 <= classical.optimality_gap(problem, index) <= 1.0


def test_local_search_never_makes_things_worse():
    problem = random_problem(8, 4, seed=11)
    start, _ = classical.field_ranking(problem)
    end, _ = classical.swap_local_search(problem, start)
    assert classical.optimality_gap(problem, end) <= classical.optimality_gap(problem, start)
