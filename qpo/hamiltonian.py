"""QUBO -> Ising, with every constant kept and a brute-force check.

The contract of this module is one identity::

    cost(x) == scale * <x| op |x> + offset

for every bitstring x (``subspace="all"``), or for every bitstring with the
right budget (``subspace="budget"``). ``verify`` checks it exhaustively.

Two reductions are available:

* ``normalise``: divide the operator by its largest coefficient so the QAOA
  angle gamma is O(1). ``scale`` remembers the factor.
* ``budget_gauge``: on the fixed-budget subspace, sum(Z) = m = n - 2B is a
  constant. Therefore, exactly and for free,

      sum_i Z_i                    = m
      sum_{i<j} (d_i + d_j) Z_i Z_j = m * sum_i d_i Z_i - sum_i d_i

  so the mean field and the whole *additive* part of the couplings can be
  moved into the offset or into single-qubit fields. What is left in the
  couplings is only the part that actually distinguishes one feasible
  portfolio from another. Valid only with a budget-preserving (XY) mixer.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from qiskit.quantum_info import SparsePauliOp

from .problem import PortfolioProblem


@dataclass(frozen=True)
class CostHamiltonian:
    op: SparsePauliOp      # sum_i h_i Z_i + sum_{i<j} J_ij Z_i Z_j  (already divided by scale)
    h: np.ndarray          # (n,)   fields of op
    J: np.ndarray          # (n,n)  upper-triangular couplings of op
    offset: float
    scale: float
    subspace: str          # "all" or "budget": where the identity holds
    lam: float

    @property
    def n(self) -> int:
        return self.h.size

    def diagonal(self) -> np.ndarray:
        """<z| op |z> for every basis state z, computed from h and J."""
        n = self.n
        z = np.arange(2 ** n)
        spins = 1.0 - 2.0 * ((z[:, None] >> np.arange(n)) & 1)   # Z eigenvalue: x=0 -> +1, x=1 -> -1
        return spins @ self.h + np.einsum("zi,ij,zj->z", spins, self.J, spins)

    def costs(self) -> np.ndarray:
        """scale * diagonal + offset: should reproduce the problem's cost."""
        return self.scale * self.diagonal() + self.offset

    @property
    def n_couplings(self) -> int:
        return int(np.count_nonzero(self.J))


# --------------------------------------------------------------------------
# QUBO and Ising coefficients
# --------------------------------------------------------------------------
def qubo_coefficients(problem: PortfolioProblem, lam: float):
    """cost_pen(x) = sum_i a_i x_i + sum_{i<j} b_ij x_i x_j + c0.

    Uses x_i^2 = x_i. The covariance is symmetric, so the pair (i, j) appears
    twice in x' S x: that is the factor 2 in b. The penalty expands to
    lam * [ (1 - 2B) sum x_i + 2 sum_{i<j} x_i x_j + B^2 ], and the B^2
    constant goes into c0.
    """
    n, B = problem.n, problem.budget
    S = problem.risk_aversion * problem.sigma / B ** 2
    a = np.diag(S) - problem.mu / B + lam * (1 - 2 * B)
    b = np.zeros((n, n))
    iu = np.triu_indices(n, 1)
    b[iu] = 2.0 * S[iu] + 2.0 * lam
    c0 = lam * B ** 2
    return a, b, c0


def ising_coefficients(a, b, c0):
    """Substitute x_i = (1 - Z_i) / 2."""
    b_sym = b + b.T
    h = -a / 2.0 - b_sym.sum(axis=1) / 4.0
    J = b / 4.0
    offset = c0 + a.sum() / 2.0 + b.sum() / 4.0
    return h, J, offset


def budget_gauge(h, J, offset, n: int, budget: int):
    """Remove everything that is constant on the fixed-budget subspace."""
    m = n - 2 * budget
    iu = np.triu_indices(n, 1)
    J_sym = J + J.T
    if n > 2:
        # least-squares additive fit J_ij ~ d_i + d_j
        d = (J_sym.sum(axis=1) - J[iu].sum() / (n - 1)) / (n - 2)
    else:
        d = np.full(n, J[iu].sum() / 2.0)
    J_new = np.zeros_like(J)
    J_new[iu] = J[iu] - (d[:, None] + d[None, :])[iu]
    h_new = h + m * d
    offset_new = offset - d.sum()
    h_mean = h_new.mean()
    return h_new - h_mean, J_new, offset_new + h_mean * m


def to_sparse_pauli_op(h, J) -> SparsePauliOp:
    n = h.size
    terms = [("Z", [i], float(h[i])) for i in range(n) if h[i] != 0.0]
    terms += [("ZZ", [i, j], float(J[i, j]))
              for i in range(n) for j in range(i + 1, n) if J[i, j] != 0.0]
    if not terms:
        terms = [("I", [0], 0.0)]
    return SparsePauliOp.from_sparse_list(terms, num_qubits=n)


# --------------------------------------------------------------------------
# Builders
# --------------------------------------------------------------------------
def build_cost_hamiltonian(problem: PortfolioProblem, lam: float = 0.0, *,
                           gauge: bool = False, normalise: bool = True,
                           drop_below: float = 0.0) -> CostHamiltonian:
    """Ising Hamiltonian for the (optionally penalised) portfolio objective.

    gauge=False: identity holds on all 2^n states (use with the X mixer and a
                 penalty lam > 0).
    gauge=True : identity holds on budget-feasible states only (use with the
                 XY mixer; lam is irrelevant there and should be 0).
    drop_below : after normalisation, zero couplings with |J| < drop_below.
                 This is an APPROXIMATION; ``verify`` reports the error.
    """
    a, b, c0 = qubo_coefficients(problem, lam)
    h, J, offset = ising_coefficients(a, b, c0)
    if gauge:
        h, J, offset = budget_gauge(h, J, offset, problem.n, problem.budget)
    scale = 1.0
    if normalise:
        scale = float(max(np.abs(h).max(), np.abs(J).max()))
        if scale == 0.0:
            scale = 1.0
        h, J = h / scale, J / scale
    if drop_below > 0.0:
        J = np.where(np.abs(J) < drop_below, 0.0, J)
    return CostHamiltonian(op=to_sparse_pauli_op(h, J), h=h, J=J, offset=float(offset),
                           scale=scale, subspace="budget" if gauge else "all", lam=lam)


def verify(problem: PortfolioProblem, ham: CostHamiltonian) -> dict:
    """Brute-force check of the Hamiltonian against the true objective.

    Compares on the states where the identity is claimed to hold, and also
    confirms that the operator's own matrix agrees with (h, J).
    """
    truth = problem.penalised_cost_vector(ham.lam)
    recon = ham.costs()
    mask = problem.feasible_mask if ham.subspace == "budget" else np.ones_like(truth, bool)
    err = np.abs(recon - truth)[mask]
    op_diag = np.real(ham.op.to_matrix(sparse=True).diagonal())
    same_argmin = int(np.flatnonzero(mask)[np.argmin(recon[mask])]) == problem.best_index \
        if ham.subspace == "budget" else None
    return {
        "max_abs_error": float(err.max()),
        "operator_matches_coefficients": bool(np.allclose(op_diag, ham.diagonal(), atol=1e-10)),
        "states_checked": int(mask.sum()),
        "feasible_argmin_preserved": same_argmin,
    }


# --------------------------------------------------------------------------
# The defects from the review, reconstructed so the check can be shown failing
# --------------------------------------------------------------------------
def build_defective_hamiltonian(problem: PortfolioProblem, lam: float) -> CostHamiltonian:
    """Reconstruction of the two bookkeeping defects found in review.

    1. The covariance cross-term reads only the upper triangle of the
       symmetric matrix (so risk coupling is half-weighted).
    2. The penalty constant lam * B^2 is never added to the offset.

    This is NOT the original code (which we do not have); it reproduces the
    defects as described so ``verify`` can be demonstrated catching them.
    """
    n, B = problem.n, problem.budget
    S = problem.risk_aversion * problem.sigma / B ** 2
    a = np.diag(S) - problem.mu / B + lam * (1 - 2 * B)
    b = np.zeros((n, n))
    iu = np.triu_indices(n, 1)
    b[iu] = 1.0 * S[iu] + 2.0 * lam          # defect 1: should be 2.0 * S
    c0 = 0.0                                 # defect 2: should be lam * B**2
    h, J, offset = ising_coefficients(a, b, c0)
    return CostHamiltonian(op=to_sparse_pauli_op(h, J), h=h, J=J, offset=float(offset),
                           scale=1.0, subspace="all", lam=lam)
