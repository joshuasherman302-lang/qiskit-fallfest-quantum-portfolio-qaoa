"""Two error-mitigation steps for sampled portfolios.

1. Readout mitigation. Measure all-|0> and all-|1> on the same physical
   qubits to estimate each qubit's flip rates, then apply the inverse of the
   per-qubit confusion matrices to the measured distribution. Assumes
   readout errors are independent between qubits. The inverse can produce
   small negative entries; they are clipped and the vector renormalised.

2. Budget post-selection (symmetry verification). The XY-mixer circuit can
   only output bitstrings with exactly ``budget`` ones. Any other outcome
   proves an error occurred during that shot, so it is discarded. This
   costs shots (the kept fraction is reported) and cannot detect errors
   that preserve the number of ones.
"""
from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit, transpile

from .hardware import sample
from .metrics import counts_to_probs


def readout_calibration(sampler_backend, physical_qubits: list, shots: int = 8192) -> np.ndarray:
    """Per-qubit flip rates. Returns (n, 2): [P(read 1 | prepared 0), P(read 0 | prepared 1)]."""
    n = len(physical_qubits)
    rates = np.zeros((n, 2))
    for prepared in (0, 1):
        circuit = QuantumCircuit(n, n)
        if prepared:
            circuit.x(range(n))
        circuit.measure(range(n), range(n))
        isa = transpile(circuit, sampler_backend, initial_layout=physical_qubits,
                        optimization_level=0)
        probs = counts_to_probs(sample(sampler_backend, isa, [], shots), n)
        z = np.arange(2 ** n)
        for q in range(n):
            p_one = probs[((z >> q) & 1) == 1].sum()
            rates[q, prepared] = p_one if prepared == 0 else 1.0 - p_one
    return rates


def mitigate_readout(probs: np.ndarray, rates: np.ndarray) -> np.ndarray:
    """Apply the inverse tensor-product confusion matrix to a probability vector."""
    n = rates.shape[0]
    tensor = np.asarray(probs, float).reshape([2] * n)      # axis 0 is qubit n-1 ... axis n-1 is qubit 0
    for q in range(n):
        e01, e10 = rates[q]
        confusion = np.array([[1 - e01, e10], [e01, 1 - e10]])   # columns: prepared 0, prepared 1
        tensor = np.moveaxis(np.tensordot(np.linalg.inv(confusion), tensor, axes=([1], [n - 1 - q])),
                             0, n - 1 - q)
    out = np.clip(tensor.reshape(-1), 0.0, None)
    return out / out.sum()


def postselect_budget(probs: np.ndarray, feasible_mask: np.ndarray):
    """Keep only budget-feasible outcomes. Returns (renormalised probs, kept fraction)."""
    kept = float(probs[feasible_mask].sum())
    out = np.where(feasible_mask, probs, 0.0)
    return (out / kept if kept > 0 else out), kept
