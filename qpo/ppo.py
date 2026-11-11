"""PPO as an *amortised* angle tuner.

A portfolio desk solves a slightly different instance every day. Tuning
QAOA angles from scratch each time costs hundreds of circuit evaluations.
Here PPO (Schulman et al. 2017, via stable-baselines3) learns, offline on a
simulator, a policy that maps a few summary features of an instance
straight to the 4 Fourier amplitudes. At deployment it costs zero circuit
evaluations, and its output can warm-start SPSA or Bayesian optimisation.

Formulation: a one-step episode (contextual bandit).
    observation : summary features of the gauge-fixed, normalised Hamiltonian
    action      : 4 Fourier amplitudes in [-1, 1], scaled by AMP_SCALE
    reward      : (mean_feasible - E) / std_feasible, i.e. how many standard
                  deviations better than a random feasible portfolio the
                  QAOA state's expected cost is. It needs no knowledge of
                  the optimum, so the same reward could be estimated on
                  hardware from samples.
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np

from .ansatz import fourier_schedule
from .fastsim import FastQAOA
from .hamiltonian import CostHamiltonian, build_cost_hamiltonian
from .problem import PortfolioProblem

AMP_SCALE = 2.0
N_FEATURES = 6


def features(ham: CostHamiltonian, budget: int) -> np.ndarray:
    n = ham.n
    upper = ham.J[np.triu_indices(n, 1)]
    return np.array([n / 16.0, budget / n, ham.h.std(), np.abs(ham.h).mean(),
                     upper.std(), np.abs(upper).max()], dtype=np.float32)


class PreparedInstance:
    """Everything the environment needs about one problem, computed once."""

    def __init__(self, problem: PortfolioProblem, p: int, init: str = "pairs",
                 drop_below: float = 0.0):
        self.problem = problem
        self.p = p
        self.ham = build_cost_hamiltonian(problem, 0.0, gauge=True, normalise=True,
                                          drop_below=drop_below)
        self.sim = FastQAOA(self.ham, "xy", problem.budget, init=init)
        feasible_energy = self.sim.diag[problem.feasible_mask]
        self.mean, self.std = float(feasible_energy.mean()), float(feasible_energy.std())
        self.obs = features(self.ham, problem.budget)

    def reward(self, amps) -> float:
        energy = self.sim.energy(*fourier_schedule(amps, self.p))
        return (self.mean - energy) / self.std


class AngleEnv(gym.Env):
    """One step per episode: see an instance, propose amplitudes, get the reward."""

    def __init__(self, instances: list, seed: int = 0, use_context: bool = True):
        super().__init__()
        self.instances = instances
        self.use_context = use_context
        self.observation_space = gym.spaces.Box(-10.0, 10.0, (N_FEATURES,), np.float32)
        self.action_space = gym.spaces.Box(-1.0, 1.0, (4,), np.float32)
        self._rng = np.random.default_rng(seed)
        self._current = None

    def _observe(self):
        return self._current.obs if self.use_context else np.zeros(N_FEATURES, np.float32)

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._current = self.instances[int(self._rng.integers(len(self.instances)))]
        return self._observe(), {}

    def step(self, action):
        reward = self._current.reward(AMP_SCALE * np.asarray(action, float))
        return self._observe(), float(reward), True, False, {}


def train_policy(instances: list, total_steps: int = 40_000, seed: int = 0,
                 use_context: bool = True, verbose: int = 0):
    """Train PPO on a pool of PreparedInstance. Returns the SB3 model."""
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv

    env = DummyVecEnv([lambda i=i: AngleEnv(instances, seed=seed * 100 + i, use_context=use_context)
                       for i in range(4)])
    model = PPO("MlpPolicy", env, n_steps=128, batch_size=128, n_epochs=10, gamma=0.0,
                gae_lambda=1.0, learning_rate=1e-3, ent_coef=0.0, seed=seed, verbose=verbose,
                policy_kwargs=dict(net_arch=[32, 32], log_std_init=-1.0), device="cpu")
    model.learn(total_timesteps=total_steps)
    return model


def propose(model, instance: PreparedInstance, use_context: bool = True) -> np.ndarray:
    """Zero-shot amplitudes for an instance: the policy mean, no circuit evaluations."""
    obs = instance.obs if use_context else np.zeros(N_FEATURES, np.float32)
    action, _ = model.predict(obs, deterministic=True)
    return AMP_SCALE * np.clip(np.asarray(action, float), -1.0, 1.0)
