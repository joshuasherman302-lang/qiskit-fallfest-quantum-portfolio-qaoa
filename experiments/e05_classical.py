"""E5. What does a laptop do to this problem?

Part A: the "hardness dial". After the budget gauge the objective splits into
        single-asset fields (solved by sorting) and residual pair
        interactions (the combinatorial part). How big is the second part,
        and how often does sorting alone already find the optimum, as risk
        aversion grows?
Part B: classical baselines against QAOA samples on the same instances.
        QAOA here is the ideal (noise-free) simulator with angles from the
        PPO policy, 1024 shots, best sampled portfolio kept. The fair
        comparison for it is "1024 uniformly random feasible portfolios".
"""
from common import RESULTS, RISK_AVERSION, ci, save

import numpy as np

from qpo import classical
from qpo.ansatz import fourier_schedule
from qpo.data import load_prices
from qpo.hamiltonian import build_cost_hamiltonian
from qpo.instances import make_pool
from qpo.ppo import PreparedInstance, propose

SHOTS = 1024
P = 4


def hardness_dial(prices) -> list:
    rows = []
    for n in (8, 12):
        for q in (1.0, 3.0, 10.0, 30.0, 100.0):
            pool = make_pool(prices, "test", 100, n, n // 2, q, seed=int(700 + n + q))
            hit_sort, hit_return, share = [], [], []
            for problem in pool:
                hit_sort.append(classical.field_ranking(problem)[0] == problem.best_index)
                hit_return.append(classical.top_return(problem)[0] == problem.best_index)
                ham = build_cost_hamiltonian(problem, 0.0, gauge=True, normalise=False)
                spins = 1.0 - 2.0 * problem.bits[problem.feasible_mask]
                field = spins @ ham.h
                pair = np.einsum("zi,ij,zj->z", spins, ham.J, spins)
                share.append(pair.var() / (field + pair).var())
            row = {"n": n, "risk_aversion": q, "instances": len(pool),
                   "sorting_finds_optimum": float(np.mean(hit_sort)),
                   "top_return_finds_optimum": float(np.mean(hit_return)),
                   "interaction_share_of_variance": ci(share)}
            rows.append(row)
            print(f"n={n:2d} q={q:5.0f}: sorting optimal {row['sorting_finds_optimum']:.2f} | "
                  f"top-return optimal {row['top_return_finds_optimum']:.2f} | "
                  f"interaction share {row['interaction_share_of_variance']['median']:.3f}")
    return rows


def baselines(prices, model) -> list:
    rows = []
    for n in (8, 12, 16):
        pool = make_pool(prices, "test", 40, n, n // 2, RISK_AVERSION, seed=900 + n)
        results = {name: {"gap": [], "evals": []} for name in (
            "top_return", "markowitz_rounding", "field_ranking", "field_ranking_plus_search",
            "simulated_annealing", "random_feasible_1024", "qaoa_ppo_1024")}
        for k, problem in enumerate(pool):
            rng = np.random.default_rng(5000 + k)
            runs = {
                "top_return": classical.top_return(problem),
                "markowitz_rounding": classical.markowitz_rounding(problem),
                "field_ranking": classical.field_ranking(problem),
                "field_ranking_plus_search": classical.field_ranking_plus_search(problem),
                "simulated_annealing": classical.simulated_annealing(problem, rng, 400),
                "random_feasible_1024": classical.random_feasible(problem, rng, SHOTS),
            }
            if model is not None:
                inst = PreparedInstance(problem, P, init="pairs")
                probs = inst.sim.probabilities(*fourier_schedule(propose(model, inst), P))
                probs = np.clip(probs, 0, None)
                sampled = np.flatnonzero(rng.multinomial(SHOTS, probs / probs.sum()))
                best = int(sampled[np.argmin(problem.cost_vector[sampled])])
                runs["qaoa_ppo_1024"] = (best, SHOTS)
            for name, (index, evals) in runs.items():
                results[name]["gap"].append(classical.optimality_gap(problem, index))
                results[name]["evals"].append(evals)
        for name, r in results.items():
            if not r["gap"]:
                continue
            gaps = np.array(r["gap"])
            row = {"n": n, "method": name, "instances": len(pool), "n_feasible": pool[0].n_feasible,
                   "finds_optimum": float(np.mean(gaps < 1e-12)),
                   "mean_gap": float(gaps.mean()), "worst_gap": float(gaps.max()),
                   "median_cost_evaluations": float(np.median(r["evals"]))}
            rows.append(row)
            print(f"n={n:2d} {name:26s}: optimum found {row['finds_optimum']:.2f} | "
                  f"mean gap {row['mean_gap']:.4f} | worst gap {row['worst_gap']:.3f} | "
                  f"evals {row['median_cost_evaluations']:.0f}")
    return rows


def main():
    prices = load_prices()
    policy_path = RESULTS / "ppo_policy_context_seed0.zip"
    model = None
    if policy_path.exists():
        from stable_baselines3 import PPO
        model = PPO.load(policy_path, device="cpu")
    else:
        print("no saved PPO policy found: run e03_tuners.py first to include the QAOA row")
    out = {"risk_aversion_for_baselines": RISK_AVERSION, "shots": SHOTS,
           "hardness_dial": hardness_dial(prices), "baselines": baselines(prices, model)}
    save("e05_classical", out)


if __name__ == "__main__":
    main()
