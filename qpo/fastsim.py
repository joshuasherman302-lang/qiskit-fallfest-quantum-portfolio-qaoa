"""A small exact state-vector simulator for the two QAOA circuits.

Why this exists: tuning needs tens of thousands of energy evaluations
(PPO training, many seeds for error bars). Rebuilding and simulating a Qiskit
circuit each time is too slow for that. This simulator applies the same
unitaries directly to a NumPy vector.

It is not trusted on its own: ``tests/test_ansatz.py`` checks that it gives
the same probabilities as Qiskit's ``Statevector`` on the real circuits from
``qpo.ansatz``. The Qiskit circuit stays the source of truth for anything
that touches a backend.
"""
from __future__ import annotations

import numpy as np

from .ansatz import chain_pairs, initial_selection
from .hamiltonian import CostHamiltonian


class FastQAOA:
    def __init__(self, ham: CostHamiltonian, mixer: str = "x", budget: int | None = None,
                 init: str = "dicke"):
        self.n = ham.n
        self.dim = 2 ** self.n
        self.mixer = mixer
        self.diag = ham.diagonal()
        if mixer == "x":
            self.psi0 = np.full(self.dim, 1.0 / np.sqrt(self.dim), dtype=complex)
        elif mixer == "xy":
            if budget is None:
                raise ValueError("the XY mixer needs the budget")
            self.psi0 = np.zeros(self.dim, dtype=complex)
            if init == "spread":
                start = sum(1 << q for q in initial_selection(self.n, budget))
                self.psi0[start] = 1.0
            elif init == "dicke":
                z = np.arange(self.dim)
                weight = ((z[:, None] >> np.arange(self.n)) & 1).sum(axis=1)
                self.psi0[weight == budget] = 1.0
                self.psi0 /= np.linalg.norm(self.psi0)
            elif init == "pairs":
                if self.n % 2 or budget * 2 != self.n:
                    raise ValueError("the 'pairs' start needs an even n and budget == n / 2")
                z = np.arange(self.dim)
                one_per_pair = np.ones(self.dim, dtype=bool)
                for a in range(0, self.n, 2):
                    one_per_pair &= (((z >> a) & 1) + ((z >> (a + 1)) & 1)) == 1
                self.psi0[one_per_pair] = 1.0
                self.psi0 /= np.linalg.norm(self.psi0)
            else:
                raise ValueError(f"unknown init {init!r}")
            z = np.arange(self.dim)
            self._pairs = []
            for layer in chain_pairs(self.n):
                for (a, b) in layer:
                    hi = z[((z >> a) & 1 == 1) & ((z >> b) & 1 == 0)]
                    lo = hi ^ (1 << a) ^ (1 << b)
                    self._pairs.append((hi, lo))
        else:
            raise ValueError(f"unknown mixer {mixer!r}")

    # ---- mixers ----------------------------------------------------------
    def _mix_x(self, psi, beta):
        c, s = np.cos(beta), -1j * np.sin(beta)
        for q in range(self.n):
            view = psi.reshape(-1, 2, 2 ** q)
            psi = (c * view + s * view[:, ::-1, :]).reshape(-1)
        return psi

    def _mix_xy(self, psi, beta):
        c, s = np.cos(beta), -1j * np.sin(beta)
        for hi, lo in self._pairs:
            a, b = psi[hi], psi[lo]
            psi[hi] = c * a + s * b
            psi[lo] = c * b + s * a
        return psi

    # ---- evolution -------------------------------------------------------
    def state(self, betas, gammas) -> np.ndarray:
        psi = self.psi0.copy()
        mix = self._mix_x if self.mixer == "x" else self._mix_xy
        for beta, gamma in zip(betas, gammas):
            psi = psi * np.exp(-1j * gamma * self.diag)
            psi = mix(psi, beta)
        return psi

    def probabilities(self, betas, gammas) -> np.ndarray:
        psi = self.state(betas, gammas)
        return (psi.real ** 2 + psi.imag ** 2)

    def energy(self, betas, gammas) -> float:
        """Exact <H> of the (normalised) cost operator."""
        return float(self.probabilities(betas, gammas) @ self.diag)
