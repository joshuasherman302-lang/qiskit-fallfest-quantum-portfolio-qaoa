"""The portfolio problem itself, with no quantum anything in it.

Everything downstream is checked against ``PortfolioProblem.cost``. If the
Hamiltonian, the simulator or the hardware run disagrees with this file,
this file wins.

Convention: asset ``i`` is qubit ``i``. A basis-state index ``z`` encodes the
selection ``x_i = (z >> i) & 1`` (Qiskit's little-endian order, so asset 0 is
the right-most character of a printed bitstring).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property
from math import comb

import numpy as np


@dataclass(frozen=True)
class PortfolioProblem:
    """Pick exactly ``budget`` of ``n`` assets, equally weighted.

    Objective (to minimise), with weights ``w = x / budget``::

        cost(x) = risk_aversion * w' Sigma w  -  mu' w

    so ``cost`` is "annual variance times risk aversion, minus annual return".
    """

    mu: np.ndarray
    sigma: np.ndarray
    budget: int
    risk_aversion: float = 3.0
    tickers: tuple = field(default=())

    def __post_init__(self):
        mu = np.asarray(self.mu, dtype=float)
        sigma = np.asarray(self.sigma, dtype=float)
        if sigma.shape != (mu.size, mu.size):
            raise ValueError("sigma must be n x n")
        if not np.allclose(sigma, sigma.T, atol=1e-12):
            raise ValueError("sigma must be symmetric")
        if not 0 < self.budget < mu.size:
            raise ValueError("budget must satisfy 0 < budget < n")
        object.__setattr__(self, "mu", mu)
        object.__setattr__(self, "sigma", sigma)
        if not self.tickers:
            object.__setattr__(self, "tickers", tuple(f"A{i}" for i in range(mu.size)))

    # ---- sizes -----------------------------------------------------------
    @property
    def n(self) -> int:
        return self.mu.size

    @property
    def n_feasible(self) -> int:
        return comb(self.n, self.budget)

    # ---- the objective ---------------------------------------------------
    def cost(self, x) -> float:
        """True objective for one selection vector x (0/1 per asset)."""
        x = np.asarray(x, dtype=float)
        w = x / self.budget
        return float(self.risk_aversion * w @ self.sigma @ w - self.mu @ w)

    def penalised_cost(self, x, lam: float) -> float:
        """Objective plus the budget penalty lam * (sum(x) - budget)^2."""
        x = np.asarray(x, dtype=float)
        return self.cost(x) + lam * (x.sum() - self.budget) ** 2

    # ---- exhaustive tables (small n only) --------------------------------
    @cached_property
    def bits(self) -> np.ndarray:
        """(2^n, n) array: row z is the selection vector of basis state z."""
        if self.n > 22:
            raise ValueError("exhaustive tables are limited to n <= 22")
        z = np.arange(2 ** self.n)
        return ((z[:, None] >> np.arange(self.n)) & 1).astype(float)

    @cached_property
    def hamming(self) -> np.ndarray:
        return self.bits.sum(axis=1).astype(int)

    @cached_property
    def feasible_mask(self) -> np.ndarray:
        return self.hamming == self.budget

    @cached_property
    def cost_vector(self) -> np.ndarray:
        """True objective for every one of the 2^n selections."""
        w = self.bits / self.budget
        return self.risk_aversion * np.einsum("zi,ij,zj->z", w, self.sigma, w) - w @ self.mu

    def penalised_cost_vector(self, lam: float) -> np.ndarray:
        return self.cost_vector + lam * (self.hamming - self.budget) ** 2

    # ---- exact answer ----------------------------------------------------
    @cached_property
    def feasible_costs(self) -> np.ndarray:
        return self.cost_vector[self.feasible_mask]

    @cached_property
    def best_index(self) -> int:
        """Basis-state index of the optimal feasible portfolio (brute force)."""
        idx = np.flatnonzero(self.feasible_mask)
        return int(idx[np.argmin(self.cost_vector[idx])])

    @property
    def best_cost(self) -> float:
        return float(self.cost_vector[self.best_index])

    @property
    def worst_feasible_cost(self) -> float:
        return float(self.feasible_costs.max())

    def selection(self, index: int) -> np.ndarray:
        return ((index >> np.arange(self.n)) & 1).astype(int)

    def names(self, index: int) -> tuple:
        return tuple(t for t, b in zip(self.tickers, self.selection(index)) if b)

    def bitstring(self, index: int) -> str:
        """Qiskit-style label: asset 0 is the right-most character."""
        return format(index, f"0{self.n}b")
