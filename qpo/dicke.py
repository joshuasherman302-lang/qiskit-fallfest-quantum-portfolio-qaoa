"""Deterministic Dicke-state preparation (Bartschi & Eidenbenz, FCT 2019).

|D(n, k)> is the equal superposition of all n-bit strings with exactly k
ones: here, every portfolio that satisfies the budget, with no preference
for any of them. Starting QAOA from it means the starting state cannot
"accidentally" be the answer, which a basis-state start can.

Construction: start from |0..0 1..1> (k ones) and apply "split and cyclic
shift" blocks. SCS(m, l) acts on the first m qubits and decides, with the
right amplitude, whether qubit m keeps its 1 or the block of ones shifts
left by one position.
"""
from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import RYGate


def _scs(circuit: QuantumCircuit, m: int, block: int) -> None:
    last = m - 1
    # two-qubit gate on (last-1, last)
    circuit.cx(last - 1, last)
    circuit.cry(2.0 * np.arccos(np.sqrt(1.0 / m)), last, last - 1)
    circuit.cx(last - 1, last)
    # three-qubit gates on (last-l, last-l+1, last)
    for l in range(2, block + 1):
        target = last - l
        circuit.cx(target, last)
        circuit.append(RYGate(2.0 * np.arccos(np.sqrt(l / m))).control(2, annotated=False),
                       [last, target + 1, target])
        circuit.cx(target, last)


def dicke_circuit(n: int, k: int) -> QuantumCircuit:
    """Circuit preparing |D(n, k)> from |0...0>."""
    if not 0 < k < n:
        raise ValueError("need 0 < k < n")
    circuit = QuantumCircuit(n, name=f"Dicke({n},{k})")
    for q in range(n - k, n):
        circuit.x(q)
    for m in range(n, k, -1):
        _scs(circuit, m, k)
    for m in range(k, 1, -1):
        _scs(circuit, m, m - 1)
    return circuit
