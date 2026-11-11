"""Angle tuners with a shared, honest evaluation budget.

Every tuner sees the same thing: a noisy objective (finite shots) that it
may call at most ``budget`` times. Calibration probes, initial designs and
warm-up samples all count. Each tuner returns the point it *recommends*;
the experiment then scores that point with the exact simulator, so nobody
is rewarded for getting a lucky noisy sample.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
from scipy.stats import norm, qmc
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

from .ansatz import fourier_schedule
from .fastsim import FastQAOA


class BudgetExhausted(Exception):
    pass


@dataclass
class Objective:
    """Shot-noise estimate of the normalised energy as a function of the
    Fourier amplitudes. Logs every call and enforces the budget."""

    sim: FastQAOA
    p: int
    shots: int | None
    rng: np.random.Generator
    budget: int
    xs: list = field(default_factory=list)
    ys: list = field(default_factory=list)

    def exact(self, amps) -> float:
        return self.sim.energy(*fourier_schedule(amps, self.p))

    def __call__(self, amps) -> float:
        if len(self.ys) >= self.budget:
            raise BudgetExhausted
        probs = self.sim.probabilities(*fourier_schedule(amps, self.p))
        if self.shots is None:
            value = float(probs @ self.sim.diag)
        else:
            probs = np.clip(probs, 0.0, None)
            counts = self.rng.multinomial(self.shots, probs / probs.sum())
            value = float(counts @ self.sim.diag / self.shots)
        self.xs.append(np.array(amps, dtype=float))
        self.ys.append(value)
        return value

    @property
    def n_evals(self) -> int:
        return len(self.ys)


# --------------------------------------------------------------------------
def random_search(objective: Objective, x0, bounds, rng):
    """Baseline every other tuner has to beat: uniform samples in the box."""
    lo, hi = bounds
    best_x, best_y = np.array(x0, float), objective(x0)
    try:
        while True:
            x = rng.uniform(lo, hi, size=len(x0))
            y = objective(x)
            if y < best_y:
                best_x, best_y = x, y
    except BudgetExhausted:
        pass
    return best_x


def spsa(objective: Objective, x0, bounds, rng, target_magnitude: float | None = None):
    """Qiskit's SPSA with its self-calibrating gains.

    The calibration (25 gradient probes = 50 evaluations) is paid out of the
    same budget. The recommendation is the last iterate, not the best noisy
    sample seen.

    target_magnitude: size of the first update step. None keeps Qiskit's
    default (2*pi/10, sized for raw angles with period 2*pi). Our parameters
    are Fourier amplitudes whose useful range is about +-1, so a smaller
    step can be set; e03b chooses it on training instances.
    """
    from qiskit_algorithms.optimizers import SPSA
    from qiskit_algorithms.utils import algorithm_globals

    algorithm_globals.random_seed = int(rng.integers(2 ** 31))
    last = {"x": np.array(x0, float)}

    def callback(nfev, x, fx, step, accepted):
        last["x"] = np.array(x, float)

    lo, hi = bounds
    fun = lambda x: objective(np.clip(x, lo, hi))
    maxiter = max((objective.budget - 50) // 2, 1)
    try:
        if target_magnitude is None:
            optimizer = SPSA(maxiter=maxiter, callback=callback)
        else:
            learning_rate, perturbation = SPSA.calibrate(
                fun, np.array(x0, float), c=target_magnitude, target_magnitude=target_magnitude)
            optimizer = SPSA(maxiter=maxiter, callback=callback, learning_rate=learning_rate,
                             perturbation=perturbation)
        result = optimizer.minimize(fun, np.array(x0, float))
        last["x"] = np.array(result.x, float)
    except BudgetExhausted:
        pass
    return np.clip(last["x"], lo, hi)


def bayes_opt(objective: Objective, x0, bounds, rng, n_init: int = 12, refit_every: int = 4,
              n_candidates: int = 3000):
    """Gaussian-process Bayesian optimisation with expected improvement.

    GP: constant * Matern-5/2 (one length scale per dimension) + white noise,
    so shot noise is modelled rather than interpolated. The recommendation is
    the evaluated point with the lowest *posterior mean*.
    """
    lo, hi = np.asarray(bounds[0], float), np.asarray(bounds[1], float)
    dim = len(x0)
    unit = lambda x: (np.asarray(x, float) - lo) / (hi - lo)
    real = lambda u: lo + u * (hi - lo)

    U, Y = [], []
    design = qmc.Sobol(dim, seed=int(rng.integers(2 ** 31))).random(n_init - 1)
    try:
        for u in [unit(x0), *design]:
            Y.append(objective(real(u)))
            U.append(u)
        kernel = (ConstantKernel(1.0, (1e-3, 1e3))
                  * Matern(length_scale=np.full(dim, 0.3), length_scale_bounds=(0.03, 3.0), nu=2.5)
                  + WhiteKernel(1e-2, (1e-6, 1e1)))
        gp = GaussianProcessRegressor(kernel, normalize_y=True, n_restarts_optimizer=1,
                                      random_state=int(rng.integers(2 ** 31)))
        step = 0
        while True:
            Ua, Ya = np.array(U), np.array(Y)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                if step % refit_every == 0:
                    gp.fit(Ua, Ya)
                else:                                   # keep hyperparameters, update the data
                    frozen = GaussianProcessRegressor(gp.kernel_, normalize_y=True, optimizer=None)
                    frozen.fit(Ua, Ya)
                    gp_step = frozen
                model = gp if step % refit_every == 0 else gp_step
                incumbent_u = Ua[np.argmin(model.predict(Ua))]
                best_mean = model.predict(incumbent_u[None, :])[0]
                cand = np.vstack([rng.random((n_candidates, dim)),
                                  np.clip(incumbent_u + rng.normal(0, 0.05, (n_candidates // 3, dim)), 0, 1)])
                mean, std = model.predict(cand, return_std=True)
            std = np.maximum(std, 1e-9)
            improvement = best_mean - mean
            ei = improvement * norm.cdf(improvement / std) + std * norm.pdf(improvement / std)
            u_next = cand[np.argmax(ei)]
            Y.append(objective(real(u_next)))
            U.append(u_next)
            step += 1
    except BudgetExhausted:
        pass
    Ua, Ya = np.array(U), np.array(Y)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        final = GaussianProcessRegressor(gp.kernel_ if hasattr(gp, "kernel_") else kernel,
                                         normalize_y=True, optimizer=None).fit(Ua, Ya)
        return real(Ua[np.argmin(final.predict(Ua))])


TUNERS = {"random": random_search, "spsa": spsa, "bayes": bayes_opt}
