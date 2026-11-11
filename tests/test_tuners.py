"""Tuners must respect the evaluation budget and return points inside the box."""
import numpy as np
import pytest

from qpo.ansatz import fourier_schedule
from qpo.hamiltonian import build_cost_hamiltonian
from qpo.fastsim import FastQAOA
from qpo.tuners import TUNERS, BudgetExhausted, Objective, spsa
from tests.test_hamiltonian import random_problem


def make_objective(budget, shots=256, seed=0):
    problem = random_problem(6, 3, seed=2)
    sim = FastQAOA(build_cost_hamiltonian(problem, 0.0, gauge=True), "xy", 3, init="pairs")
    return Objective(sim, 2, shots, np.random.default_rng(seed), budget)


def test_objective_stops_at_the_budget():
    objective = make_objective(budget=3)
    for _ in range(3):
        objective(np.zeros(4))
    with pytest.raises(BudgetExhausted):
        objective(np.zeros(4))


def test_shot_noise_averages_to_the_exact_energy():
    objective = make_objective(budget=400, shots=512)
    amps = np.array([0.6, -0.1, -0.5, 0.2])
    noisy = np.mean([objective(amps) for _ in range(400)])
    assert noisy == pytest.approx(objective.exact(amps), abs=0.02)
    assert objective.exact(amps) == pytest.approx(objective.sim.energy(*fourier_schedule(amps, 2)))


@pytest.mark.parametrize("name", sorted(TUNERS))
def test_tuner_respects_budget_and_bounds(name):
    budget = 70
    objective = make_objective(budget)
    bounds = (np.full(4, -2.0), np.full(4, 2.0))
    x = TUNERS[name](objective, np.zeros(4), bounds, np.random.default_rng(1))
    assert objective.n_evals <= budget
    assert objective.n_evals >= budget - 2          # the budget is actually used
    assert x.shape == (4,) and (x >= -2.0).all() and (x <= 2.0).all()


def test_spsa_with_a_set_step_size_respects_the_budget():
    objective = make_objective(90)
    bounds = (np.full(4, -2.0), np.full(4, 2.0))
    x = spsa(objective, np.full(4, 0.2), bounds, np.random.default_rng(3), target_magnitude=0.1)
    assert objective.n_evals <= 90
    assert (np.abs(x) <= 2.0).all()
