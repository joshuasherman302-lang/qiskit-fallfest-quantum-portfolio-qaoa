"""E4. Does it survive ibm_brisbane's connectivity and noise?

Target: FakeBrisbane (127 qubits, heavy-hex, ECR/RZ/SX/X, calibration
snapshot). The real device was retired on 2025-11-03, so everything here is
a noisy SIMULATION with that device's noise model, not a hardware run.

Part A: two-qubit gate cost of each building block after transpilation.
Part B: noisy sampling of whole circuits, raw and mitigated.
"""
from common import RISK_AVERSION, ceiling_angles, ci, save

import sys
import time

import numpy as np
from qiskit import QuantumCircuit

from qpo.ansatz import append_initial_state, bind_vector, build_qaoa_circuit
from qpo.data import load_prices
from qpo.fastsim import FastQAOA
from qpo.hamiltonian import build_cost_hamiltonian
from qpo.hardware import (best_line, build_swap_network_circuit, get_backend,
                          measured_physical_qubits, noisy_simulator, sample, transpile_for,
                          two_qubit_stats)
from qpo.instances import make_pool
from qpo.metrics import counts_to_probs, summarise
from qpo.mitigation import mitigate_readout, postselect_budget, readout_calibration

SHOTS = 4096
FIELDS_ONLY = dict(drop_below=10.0)      # after normalisation every |J| <= 1, so this drops all couplings


def gate_costs(backend, prices) -> list:
    rows = []
    for n in (4, 6, 8, 12, 16):
        budget = n // 2
        problem = make_pool(prices, "test", 1, n, budget, RISK_AVERSION, seed=n)[0]
        full = build_cost_hamiltonian(problem, 0.0, gauge=True)
        fields = build_cost_hamiltonian(problem, 0.0, gauge=True, **FIELDS_ONLY)
        line = best_line(backend, n)
        values = [0.37, 0.81]

        def prep(init):
            circuit = QuantumCircuit(n)
            append_initial_state(circuit, n, budget, init)
            return two_qubit_stats(transpile_for(backend, circuit, initial_layout=line))

        naive_circuit = build_qaoa_circuit(full, 1, "xy", budget, init="spread", measure=True)
        naive = [two_qubit_stats(transpile_for(backend, naive_circuit.assign_parameters(values), seed=s))
                 for s in range(5)]
        swap = two_qubit_stats(transpile_for(
            backend, build_swap_network_circuit(full, 1, budget, init="spread").assign_parameters(values),
            initial_layout=line))
        fld = two_qubit_stats(transpile_for(
            backend, build_qaoa_circuit(fields, 1, "xy", budget, init="spread", measure=True)
            .assign_parameters(values), initial_layout=line))
        row = {
            "n": n, "line": line,
            "prep_dicke": prep("dicke"), "prep_pairs": prep("pairs"),
            "layer_naive_median": {k: int(np.median([s[k] for s in naive])) for k in naive[0]},
            "layer_naive_best": {k: int(min(s[k] for s in naive)) for k in naive[0]},
            "layer_swap_network": swap, "layer_fields_only": fld,
        }
        rows.append(row)
        g = lambda s: f"{s['two_qubit_gates']:4d}/{s['two_qubit_depth']:3d}"
        print(f"n={n:2d} | dicke {g(row['prep_dicke'])} | pairs {g(row['prep_pairs'])} | "
              f"naive {g(row['layer_naive_median'])} | swap-net {g(swap)} | fields-only {g(fld)}")
    return rows


DESIGNS = [
    # name, mixer, init, hamiltonian kwargs, p, layout
    ("x_penalty p=1",          "x",  None,     "penalty", 1, "naive"),
    ("dicke + full H p=1",     "xy", "dicke",  {},        1, "swap"),
    ("pairs + full H p=1",     "xy", "pairs",  {},        1, "swap"),
    ("pairs + full H p=2",     "xy", "pairs",  {},        2, "swap"),
    ("pairs + fields only p=2", "xy", "pairs", FIELDS_ONLY, 2, "line"),
    ("pairs + fields only p=4", "xy", "pairs", FIELDS_ONLY, 4, "line"),
]


def noisy_runs(backend, prices, sizes, n_instances) -> list:
    simulator = noisy_simulator(backend, seed=11)
    rng = np.random.default_rng(404)
    calibrations = {}
    cells = []
    for n in sizes:
        budget = n // 2
        pool = make_pool(prices, "test", n_instances, n, budget, RISK_AVERSION, seed=300 + n)
        line = best_line(backend, n)
        for name, mixer, init, ham_kw, p, layout in DESIGNS:
            rows, t0 = [], time.time()
            for problem in pool:
                if ham_kw == "penalty":
                    ham = build_cost_hamiltonian(problem, lam=problem.worst_feasible_cost - problem.best_cost)
                    sim = FastQAOA(ham, "x")
                else:
                    ham = build_cost_hamiltonian(problem, 0.0, gauge=True, **ham_kw)
                    sim = FastQAOA(ham, "xy", budget, init)
                (betas, gammas), _ = ceiling_angles(sim, p, 6, rng)
                ideal = summarise(problem, sim.probabilities(betas, gammas))

                if layout == "swap":
                    circuit = build_swap_network_circuit(ham, p, budget, init=init)
                    isa = transpile_for(backend, circuit, initial_layout=line)
                elif layout == "line":
                    circuit = build_qaoa_circuit(ham, p, mixer, budget, init=init, measure=True)
                    isa = transpile_for(backend, circuit, initial_layout=line)
                else:
                    circuit = build_qaoa_circuit(ham, p, mixer, budget, measure=True)
                    isa = transpile_for(backend, circuit, seed=0)
                stats = two_qubit_stats(isa)

                raw = counts_to_probs(sample(simulator, isa, bind_vector(betas, gammas), SHOTS), n)
                qubits = tuple(measured_physical_qubits(isa))
                if qubits not in calibrations:
                    calibrations[qubits] = readout_calibration(simulator, list(qubits))
                mitigated = mitigate_readout(raw, calibrations[qubits])

                raw_post, raw_kept = postselect_budget(raw, problem.feasible_mask)
                mit_post, mit_kept = postselect_budget(mitigated, problem.feasible_mask)
                rows.append({
                    "ideal_ratio": ideal["ratio"], "ideal_enhancement": ideal["enhancement"],
                    "ideal_p_feasible": ideal["p_feasible"],
                    "raw_p_feasible": raw_kept,
                    "raw_ratio": summarise(problem, raw_post)["ratio"],
                    "raw_enhancement": summarise(problem, raw_post)["enhancement"],
                    "mitigated_p_feasible": mit_kept,
                    "mitigated_ratio": summarise(problem, mit_post)["ratio"],
                    "mitigated_enhancement": summarise(problem, mit_post)["enhancement"],
                    "ratio_uniform": ideal["ratio_uniform"],
                    **stats,
                })
            cell = {"n": n, "design": name, "p": p,
                    "two_qubit_gates": int(np.median([r["two_qubit_gates"] for r in rows])),
                    "two_qubit_depth": int(np.median([r["two_qubit_depth"] for r in rows])),
                    "ratio_uniform": float(np.median([r["ratio_uniform"] for r in rows]))}
            for key in ("ideal_ratio", "ideal_enhancement", "ideal_p_feasible", "raw_p_feasible",
                        "raw_ratio", "raw_enhancement", "mitigated_p_feasible", "mitigated_ratio",
                        "mitigated_enhancement"):
                cell[key] = ci([r[key] for r in rows])
            cells.append(cell)
            m = lambda k: cell[k]["median"]
            print(f"n={n} {name:24s} 2q={cell['two_qubit_gates']:4d} | feasible: ideal {m('ideal_p_feasible'):.2f} "
                  f"raw {m('raw_p_feasible'):.2f} | ratio: ideal {m('ideal_ratio'):.3f} raw+postsel {m('raw_ratio'):.3f} "
                  f"mitigated {m('mitigated_ratio'):.3f} (guess {cell['ratio_uniform']:.3f}) | "
                  f"P(best|f): ideal {m('ideal_enhancement'):.1f}x mitigated {m('mitigated_enhancement'):.1f}x "
                  f"[{time.time() - t0:.0f}s]")
            sys.stdout.flush()
    readout = np.concatenate([c.reshape(-1) for c in calibrations.values()])
    return cells, float(np.median(readout))


def main():
    quick = "--quick" in sys.argv
    prices = load_prices()
    backend = get_backend("fake_brisbane")
    ecr = [props.error for props in backend.target["ecr"].values() if props and props.error]
    out = {"backend": "FakeBrisbane (calibration snapshot of ibm_brisbane, retired 2025-11-03)",
           "shots": SHOTS, "risk_aversion": RISK_AVERSION,
           "median_ecr_error": float(np.median(ecr))}
    print(f"median ECR error on the snapshot: {out['median_ecr_error']:.4f}")
    out["gate_costs"] = gate_costs(backend, prices)
    cells, readout = noisy_runs(backend, prices, (4,) if quick else (4, 6, 8), 2 if quick else 5)
    out["noisy"] = cells
    out["median_readout_flip_rate"] = readout
    print(f"median calibrated readout flip rate: {readout:.4f}")
    save("e04_hardware_quick" if quick else "e04_hardware", out)


if __name__ == "__main__":
    main()
