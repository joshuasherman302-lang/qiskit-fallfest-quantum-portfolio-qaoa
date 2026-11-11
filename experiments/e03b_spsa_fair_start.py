"""E3b. A fairness check for SPSA. Run after e03_tuners.py (it loads the saved PPO policy).

E3 runs Qiskit's SPSA exactly as shipped, and it does badly. Two things are
stacked against it there, neither of which is SPSA's fault:

1. Every tuner in E3 starts from all-zero amplitudes, a stationary point of
   this circuit. SPSA's self-calibration divides by the measured gradient.
2. Qiskit's default first step is 2*pi/10, sized for raw angles. Our
   parameters are Fourier amplitudes with a useful range of about +-1.

So here SPSA gets a small random start (uniform in [-0.5, 0.5]) and a step
size chosen from {0.05, 0.1, 0.2} on TRAINING instances only. It is then run
on the same held-out instances and budgets as E3, from the random start and
from the PPO proposal. Rows are appended to results/e03_tuners.json.
"""
from common import RESULTS, RISK_AVERSION, ci

import json

import numpy as np

from qpo.data import load_prices
from qpo.instances import make_pool
from qpo.ppo import PreparedInstance, propose
from qpo.tuners import Objective, spsa

from e03_tuners import BOUNDS, BUDGET, EVAL_BUDGETS, N, N_TEST, P, SHOTS, score

CANDIDATES = (0.05, 0.1, 0.2)


def run(instances, budget, magnitude, seed_base, start_fn):
    rows = []
    for k, inst in enumerate(instances):
        rng = np.random.default_rng(seed_base + 97 * k + budget)
        objective = Objective(inst.sim, P, SHOTS, rng, budget)
        rows.append(score(inst, spsa(objective, start_fn(inst, rng), BOUNDS, rng,
                                     target_magnitude=magnitude)))
    return rows


def main():
    from stable_baselines3 import PPO

    prices = load_prices()
    prep = lambda pool: [PreparedInstance(pr, P, init="pairs") for pr in pool]
    train = prep(make_pool(prices, "train", 12, N, BUDGET, RISK_AVERSION, seed=77))
    test = prep(make_pool(prices, "test", N_TEST, N, BUDGET, RISK_AVERSION, seed=20261111))
    model = PPO.load(RESULTS / "ppo_policy_context_seed0.zip", device="cpu")
    random_start = lambda inst, rng: rng.uniform(-0.5, 0.5, 4)
    ppo_start = lambda inst, rng: propose(model, inst)

    selection = {m: float(np.median([r["ratio"] for r in run(train, 200, m, 40_000, random_start)]))
                 for m in CANDIDATES}
    chosen = max(selection, key=selection.get)
    print("step size chosen on training instances:", selection, "->", chosen)

    path = RESULTS / "e03_tuners.json"
    out = json.loads(path.read_text())
    out["tuners"] = [t for t in out["tuners"] if t["tuner"] != "spsa_tuned"]
    out["spsa_step_selection_on_train"] = {"candidates": selection, "chosen": chosen}
    for start_name, start_fn, budgets in (("random_small", random_start, EVAL_BUDGETS),
                                          ("ppo", ppo_start, (100,))):
        for budget in budgets:
            rows = run(test, budget, chosen, 30_000, start_fn)
            entry = {"tuner": "spsa_tuned", "start": start_name, "evals": budget,
                     "ratio": ci([r["ratio"] for r in rows]),
                     "enhancement": ci([r["enhancement"] for r in rows])}
            out["tuners"].append(entry)
            print(f"spsa (step {chosen}) from {start_name}, {budget} evals: ratio "
                  f"{entry['ratio']['median']:.3f} [{entry['ratio']['lo']:.3f}, {entry['ratio']['hi']:.3f}]")
    path.write_text(json.dumps(out, indent=2, default=float))
    print("appended to results/e03_tuners.json")


if __name__ == "__main__":
    main()
