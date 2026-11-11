"""QAOA circuits (Qiskit) and the Fourier angle schedule.

Two ansatz families:

* ``mixer="x"``  : textbook QAOA from ``qiskit.circuit.library.qaoa_ansatz``.
                   Starts in |+>^n, mixes with sum X. The budget is enforced
                   only softly, through the penalty inside the Hamiltonian.
* ``mixer="xy"`` : starts in a basis state that already holds exactly
                   ``budget`` assets and mixes with XX+YY rotations on
                   neighbouring qubits. Every gate conserves the number of
                   selected assets, so the budget holds by construction and
                   no penalty is needed.

Both circuits expose their parameters in the same order: beta[0..p-1] then
gamma[0..p-1]. ``bind_vector`` builds that vector, and the builders assert
the order so a silent swap of beta and gamma cannot happen.
"""
from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import XXPlusYYGate, qaoa_ansatz

from .dicke import dicke_circuit
from .hamiltonian import CostHamiltonian


# --------------------------------------------------------------------------
# Angle schedules
# --------------------------------------------------------------------------
def fourier_schedule(amps, p: int):
    """Zhou et al. (PRX 10, 021067) FOURIER parameterisation.

    amps = [u_1..u_q, v_1..v_q]. For layer l = 1..p:

        gamma_l = sum_k u_k sin((k - 1/2)(l - 1/2) pi / p)
        beta_l  = sum_k v_k cos((k - 1/2)(l - 1/2) pi / p)

    Returns (betas, gammas), each of length p.
    """
    amps = np.asarray(amps, dtype=float)
    if amps.size % 2:
        raise ValueError("need an even number of amplitudes: [u..., v...]")
    q = amps.size // 2
    u, v = amps[:q], amps[q:]
    layer = np.arange(p) + 0.5          # (l - 1/2) for l = 1..p
    freq = np.arange(q) + 0.5           # (k - 1/2) for k = 1..q
    phase = np.outer(layer, freq) * np.pi / p
    return np.cos(phase) @ v, np.sin(phase) @ u


def bind_vector(betas, gammas) -> np.ndarray:
    """Parameter values in circuit order: all betas, then all gammas."""
    return np.concatenate([np.asarray(betas, float), np.asarray(gammas, float)])


def _assert_beta_then_gamma(circuit: QuantumCircuit, p: int) -> None:
    names = [param.name for param in circuit.parameters]
    expected = [f"β[{i}]" for i in range(p)] + [f"γ[{i}]" for i in range(p)]
    if names != expected:
        raise AssertionError(f"unexpected parameter order: {names}")


# --------------------------------------------------------------------------
# Budget-preserving pieces
# --------------------------------------------------------------------------
def initial_selection(n: int, budget: int) -> list:
    """Qubits set to |1> in the starting state: ``budget`` ones spread evenly
    (0101... at half filling) so every neighbouring pair can exchange."""
    return sorted({int(np.floor((k + 0.5) * n / budget)) for k in range(budget)})


def chain_pairs(n: int) -> list:
    """Nearest-neighbour pairs on a line, split into two layers of disjoint
    gates: (0,1),(2,3),... then (1,2),(3,4),..."""
    even = [(i, i + 1) for i in range(0, n - 1, 2)]
    odd = [(i, i + 1) for i in range(1, n - 1, 2)]
    return [layer for layer in (even, odd) if layer]


def append_initial_state(circuit: QuantumCircuit, n: int, budget: int, init: str) -> None:
    """Budget-feasible starting states for the XY mixer.

    "dicke"  : equal superposition of every feasible portfolio. Unbiased, but
               expensive to prepare on a sparse device.
    "pairs"  : qubits (0,1), (2,3), ... each in (|01> + |10>)/sqrt(2): one
               asset per pair, either one with equal amplitude. Unbiased for
               every asset, 2^(n/2) feasible states, one CX per pair.
               Requires budget == n / 2.
    "spread" : a single basis state. Free, but it IS a candidate answer, so
               results depend on which state was picked.
    """
    if init == "dicke":
        circuit.compose(dicke_circuit(n, budget), qubits=range(n), inplace=True)
    elif init == "pairs":
        if n % 2 or budget * 2 != n:
            raise ValueError("the 'pairs' start needs an even n and budget == n / 2")
        for a in range(0, n, 2):
            circuit.h(a)
            circuit.cx(a, a + 1)
            circuit.x(a + 1)
    elif init == "spread":
        for q in initial_selection(n, budget):
            circuit.x(q)
    else:
        raise ValueError(f"unknown init {init!r}")


def append_cost_layer(circuit: QuantumCircuit, ham: CostHamiltonian, gamma) -> None:
    """exp(-i gamma H) for H = sum h_i Z_i + sum J_ij Z_i Z_j, gate by gate."""
    for i in range(ham.n):
        if ham.h[i] != 0.0:
            circuit.rz(2.0 * gamma * float(ham.h[i]), i)
    for i in range(ham.n):
        for j in range(i + 1, ham.n):
            if ham.J[i, j] != 0.0:
                circuit.rzz(2.0 * gamma * float(ham.J[i, j]), i, j)


# --------------------------------------------------------------------------
# Circuit builders
# --------------------------------------------------------------------------
def build_qaoa_circuit(ham: CostHamiltonian, p: int, mixer: str = "x", budget: int | None = None,
                       init: str = "dicke", measure: bool = False) -> QuantumCircuit:
    """Parameterised QAOA circuit on logical qubits (all-to-all connectivity).

    init (XY mixer only): "dicke", "pairs" or "spread"; see ``append_initial_state``.
    """
    if mixer == "x":
        circuit = qaoa_ansatz(ham.op, reps=p)
    elif mixer == "xy":
        if budget is None:
            raise ValueError("the XY mixer needs the budget")
        n = ham.n
        beta = ParameterVector("β", p)
        gamma = ParameterVector("γ", p)
        circuit = QuantumCircuit(n, name="QAOA-XY")
        append_initial_state(circuit, n, budget, init)
        for layer in range(p):
            append_cost_layer(circuit, ham, gamma[layer])
            for pairs in chain_pairs(n):
                for (a, b) in pairs:
                    # exp(-i beta (XX + YY) / 2)
                    circuit.append(XXPlusYYGate(2.0 * beta[layer]), [a, b])
    else:
        raise ValueError(f"unknown mixer {mixer!r}")
    _assert_beta_then_gamma(circuit, p)
    if measure:
        circuit.measure_all()
    return circuit
