"""E3. Tuning the angles: random search, SPSA, Bayesian optimisation, PPO.

Fixed design: XY mixer, paired start, budget-gauged Hamiltonian, 4 Fourier
amplitudes expanded to p = 4 layers, n = 8 assets, budget 4.

Rules that keep the comparison honest
  * Train/test split by time: PPO only sees 2021-2023 windows; every number
    reported here is on 2024-2025 windows.
  * Per-instance tuners get a noisy objective (1024 shots) and a hard cap on
    evaluations; calibration and initial designs count against the cap.
  * Every recommended point is scored with the exact simulator.
  * PPO's offline training cost is reported, not hidden.
  * PPO is trained with 5 seeds; tuners run once on each of 20 instances.
"""
from common import RESULTS, RISK_AVERSION, ceiling_angles, ci, save

import time

import numpy as np

from qpo.ansatz import fourier_schedule
from qpo.data import load_prices
from qpo.instances import make_pool
from qpo.metrics import summarise
from qpo.ppo import AMP_SCALE, PreparedInstance, propose, train_policy
from qpo.tuners import TUNERS, Objective

P, N, BUDGET = 4, 8, 4
SHOTS = 1024
EVAL_BUDGETS = (100, 200)
PPO_STEPS = 30_000
PPO_SEEDS = (0, 1, 2, 3, 4)
N_TRAIN, N_TEST = 48, 20
BOUNDS = (np.full(4, -AMP_SCALE), np.full(4, AMP_SCALE))


def score(instance: PreparedInstance, amps) -> dict:
    return summarise(instance.problem, instance.sim.probabilities(*fourier_schedule(amps, instance.p)))


def cell(rows) -> dict:
    return {"ratio": ci([r["ratio"] for r in rows]),
            "enhancement": ci([r["enhancement"] for r in rows])}


def main():
    prices = load_prices()
    prep = lambda problems: [PreparedInstance(pr, P, init="pairs") for pr in problems]
    train = prep(make_pool(prices, "train", N_TRAIN, N, BUDGET, RISK_AVERSION, seed=1))
    test = prep(make_pool(prices, "test", N_TEST, N, BUDGET, RISK_AVERSION, seed=20261111))
    out = {"config": dict(p=P, n=N, budget=BUDGET, shots=SHOTS, risk_aversion=RISK_AVERSION,
                          n_train=N_TRAIN, n_test=N_TEST, ppo_steps=PPO_STEPS,
                          ppo_seeds=list(PPO_SEEDS), amp_bounds=AMP_SCALE)}

    # ---- references ------------------------------------------------------
    rng = np.random.default_rng(7)
    neutral = np.zeros(4)
    out["uniform_guess_ratio"] = float(np.median([score(i, neutral)["ratio_uniform"] for i in test]))
    out["neutral_start"] = cell([score(i, neutral) for i in test])
    ceilings = [score(i, ceiling_angles(i.sim, P, 8, rng, fourier=True)[1]) for i in test]
    out["ceiling_fourier4"] = cell(ceilings)
    print(f"guess {out['uniform_guess_ratio']:.3f} | neutral start {out['neutral_start']['ratio']['median']:.3f} "
          f"| ceiling {out['ceiling_fourier4']['ratio']['median']:.3f}")

    # ---- PPO: train offline, propose zero-shot ---------------------------
    out["ppo"] = {}
    models = {}
    for use_context in (True, False):
        key = "context" if use_context else "no_context"
        per_seed, seconds = [], []
        for seed in PPO_SEEDS:
            t0 = time.time()
            model = train_policy(train, PPO_STEPS, seed=seed, use_context=use_context)
            seconds.append(time.time() - t0)
            rows = [score(i, propose(model, i, use_context)) for i in test]
            per_seed.append({"seed": seed,
                             "ratio_median": float(np.median([r["ratio"] for r in rows])),
                             "enhancement_median": float(np.median([r["enhancement"] for r in rows])),
                             "ratios": [r["ratio"] for r in rows]})
            models[(key, seed)] = model
            print(f"PPO {key} seed {seed}: test ratio {per_seed[-1]['ratio_median']:.3f} "
                  f"P(best|f) {per_seed[-1]['enhancement_median']:.1f}x  ({seconds[-1]:.0f}s)")
        medians = [s["ratio_median"] for s in per_seed]
        out["ppo"][key] = {"per_seed": per_seed, "train_seconds_mean": float(np.mean(seconds)),
                           "ratio_median_over_seeds": {"mean": float(np.mean(medians)),
                                                       "min": float(min(medians)),
                                                       "max": float(max(medians))}}

    models[("context", 0)].save(RESULTS / "ppo_policy_context_seed0.zip")   # reused by e05

    # ---- per-instance tuners from a neutral start ------------------------
    out["tuners"] = []
    for budget in EVAL_BUDGETS:
        for name, tuner in TUNERS.items():
            rows, t0 = [], time.time()
            for k, inst in enumerate(test):
                run_rng = np.random.default_rng(10_000 + 97 * k + budget)
                objective = Objective(inst.sim, P, SHOTS, run_rng, budget)
                rows.append(score(inst, tuner(objective, neutral, BOUNDS, run_rng)))
                assert objective.n_evals <= budget
            entry = {"tuner": name, "start": "neutral", "evals": budget,
                     "seconds_per_run": (time.time() - t0) / len(test), **cell(rows)}
            out["tuners"].append(entry)
            print(f"{name:7s} neutral start, {budget:3d} evals: ratio {entry['ratio']['median']:.3f} "
                  f"[{entry['ratio']['lo']:.3f}, {entry['ratio']['hi']:.3f}]  "
                  f"P(best|f) {entry['enhancement']['median']:.1f}x")

    # ---- warm start from the PPO proposal --------------------------------
    warm_model = models[("context", 0)]
    for name in ("spsa", "bayes"):
        rows = []
        for k, inst in enumerate(test):
            run_rng = np.random.default_rng(20_000 + 97 * k)
            objective = Objective(inst.sim, P, SHOTS, run_rng, 100)
            rows.append(score(inst, TUNERS[name](objective, propose(warm_model, inst), BOUNDS, run_rng)))
        entry = {"tuner": name, "start": "ppo", "evals": 100, **cell(rows)}
        out["tuners"].append(entry)
        print(f"{name:7s} PPO warm start, 100 evals: ratio {entry['ratio']['median']:.3f} "
              f"[{entry['ratio']['lo']:.3f}, {entry['ratio']['hi']:.3f}]")

    # ---- does the n = 8 policy transfer to larger problems? --------------
    out["transfer"] = []
    for n in (10, 12):
        pool = prep(make_pool(prices, "test", N_TEST, n, n // 2, RISK_AVERSION, seed=50 + n))
        ceil = cell([score(i, ceiling_angles(i.sim, P, 6, rng, fourier=True)[1]) for i in pool])
        entry = {"n": n, "ceiling_fourier4": ceil,
                 "uniform_guess_ratio": float(np.median([score(i, neutral)["ratio_uniform"] for i in pool])),
                 "neutral_start": cell([score(i, neutral) for i in pool])}
        for key, use_context in (("context", True), ("no_context", False)):
            medians = [float(np.median([score(i, propose(models[(key, s)], i, use_context))["ratio"]
                                        for i in pool])) for s in PPO_SEEDS]
            entry[f"ppo_{key}"] = {"mean": float(np.mean(medians)), "min": float(min(medians)),
                                   "max": float(max(medians))}
        out["transfer"].append(entry)
        print(f"transfer to n={n}: ceiling {ceil['ratio']['median']:.3f} | "
              f"PPO context {entry['ppo_context']['mean']:.3f} [{entry['ppo_context']['min']:.3f}, {entry['ppo_context']['max']:.3f}] | "
              f"PPO no-context {entry['ppo_no_context']['mean']:.3f} | guess {entry['uniform_guess_ratio']:.3f}")

    save("e03_tuners", out)


if __name__ == "__main__":
    main()
