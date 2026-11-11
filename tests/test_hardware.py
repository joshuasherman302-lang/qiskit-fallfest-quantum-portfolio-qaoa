"""Swap-network equivalence, transpilation to FakeBrisbane, and mitigation."""
import numpy as np
import pytest
from qiskit.quantum_info import Statevector

from qpo.ansatz import bind_vector, build_qaoa_circuit
from qpo.hamiltonian import build_cost_hamiltonian
from qpo.hardware import (build_swap_network_circuit, get_backend, measured_physical_qubits,
                          transpile_for, two_qubit_stats)
from qpo.mitigation import mitigate_readout, postselect_budget
from tests.test_hamiltonian import random_problem


@pytest.mark.parametrize("n,budget", [(4, 2), (5, 2), (6, 3), (7, 3)])
@pytest.mark.parametrize("p", [1, 2, 3])
@pytest.mark.parametrize("init", ["dicke", "spread"])
def test_swap_network_equals_the_logical_circuit(n, budget, p, init):
    """Same output distribution as the all-to-all circuit once positions are
    mapped back to assets."""
    rng = np.random.default_rng(n * 10 + p)
    problem = random_problem(n, budget, seed=n)
    ham = build_cost_hamiltonian(problem, 0.0, gauge=True, normalise=True)
    values = bind_vector(rng.uniform(-1, 1, p), rng.uniform(-1, 1, p))
    logical = Statevector(build_qaoa_circuit(ham, p, "xy", budget, init=init)
                          .assign_parameters(values)).probabilities()
    physical_circuit = build_swap_network_circuit(ham, p, budget, init=init, measure=False)
    physical = Statevector(physical_circuit.assign_parameters(values)).probabilities()
    order = physical_circuit.metadata["final_order"]          # order[k] = asset at position k
    z = np.arange(2 ** n)
    asset_index = np.zeros_like(z)
    for k in range(n):
        asset_index |= ((z >> k) & 1) << order[k]
    remapped = np.zeros_like(physical)
    np.add.at(remapped, asset_index, physical)
    assert np.allclose(remapped, logical, atol=1e-10)


def test_swap_network_reverses_the_line_each_layer():
    problem = random_problem(6, 3, seed=1)
    ham = build_cost_hamiltonian(problem, 0.0, gauge=True)
    assert build_swap_network_circuit(ham, 1, 3).metadata["final_order"] == [5, 4, 3, 2, 1, 0]
    assert build_swap_network_circuit(ham, 2, 3).metadata["final_order"] == [0, 1, 2, 3, 4, 5]


def test_transpiled_circuit_uses_only_brisbane_gates_and_edges():
    backend = get_backend("fake_brisbane")
    problem = random_problem(5, 2, seed=4)
    ham = build_cost_hamiltonian(problem, 0.0, gauge=True)
    isa = transpile_for(backend, build_swap_network_circuit(ham, 1, 2), seed=0)
    allowed = set(backend.target.operation_names)
    edges = {tuple(e) for e in backend.coupling_map.get_edges()}
    for inst in isa.data:
        assert inst.operation.name in allowed
        if inst.operation.num_qubits == 2:
            pair = tuple(isa.find_bit(q).index for q in inst.qubits)
            assert pair in edges or pair[::-1] in edges
    stats = two_qubit_stats(isa)
    assert stats["two_qubit_gates"] > 0
    assert len(set(measured_physical_qubits(isa))) == 5


def test_readout_mitigation_inverts_known_noise():
    rng = np.random.default_rng(0)
    n = 4
    truth = rng.dirichlet(np.ones(2 ** n))
    rates = rng.uniform(0.01, 0.08, (n, 2))
    noisy = truth.reshape([2] * n)
    for q in range(n):
        e01, e10 = rates[q]
        confusion = np.array([[1 - e01, e10], [e01, 1 - e10]])
        noisy = np.moveaxis(np.tensordot(confusion, noisy, axes=([1], [n - 1 - q])), 0, n - 1 - q)
    recovered = mitigate_readout(noisy.reshape(-1), rates)
    assert np.allclose(recovered, truth, atol=1e-10)


def test_postselection_keeps_only_the_budget():
    problem = random_problem(5, 2, seed=0)
    probs = np.random.default_rng(1).dirichlet(np.ones(32))
    kept_probs, kept = postselect_budget(probs, problem.feasible_mask)
    assert kept == pytest.approx(probs[problem.feasible_mask].sum())
    assert kept_probs[~problem.feasible_mask].sum() == 0.0
    assert kept_probs.sum() == pytest.approx(1.0)
