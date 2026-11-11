"""Getting the circuits onto a heavy-hex device (ibm_brisbane's layout).

ibm_brisbane itself was retired on 2025-11-03, so the default target is
``FakeBrisbane``: IBM's snapshot of that device's coupling map, basis gates
(ECR, RZ, SX, X) and calibration data. Any live backend name can be passed
instead; the code path is the same.

Three ways to lay out one QAOA layer:

* naive        : hand the all-to-all circuit to the transpiler (SABRE routing).
* swap network : put the assets on a line of qubits (heavy-hex contains long
                 lines) and interleave the ZZ rotations with SWAPs so that
                 every pair becomes adjacent exactly once. n layers, no
                 routing overhead beyond that, and each RZZ+SWAP pair
                 compiles to 3 ECR gates.
* fields only  : after the budget gauge, drop the (small) residual ZZ terms.
                 The cost layer is then RZ only (virtual, error-free on IBM
                 hardware) and all entangling gates are the nearest-neighbour
                 mixer, which fits a line with no SWAPs at all.
"""
from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import XXPlusYYGate
from qiskit.transpiler import generate_preset_pass_manager

from .ansatz import _assert_beta_then_gamma, append_initial_state, chain_pairs
from .hamiltonian import CostHamiltonian


# --------------------------------------------------------------------------
# Backends
# --------------------------------------------------------------------------
def get_backend(name: str = "fake_brisbane"):
    """FakeBrisbane by default; otherwise a live backend from the saved account.

    Credentials are never passed in code: save them once with
    ``QiskitRuntimeService.save_account`` outside this repository.
    """
    if name == "fake_brisbane":
        from qiskit_ibm_runtime.fake_provider import FakeBrisbane
        return FakeBrisbane()
    from qiskit_ibm_runtime import QiskitRuntimeService
    service = QiskitRuntimeService()
    if name == "least_busy":
        return service.least_busy(operational=True, simulator=False)
    return service.backend(name)


def noisy_simulator(backend, seed: int = 0):
    """Aer simulator carrying the backend's noise model and coupling map."""
    from qiskit_aer import AerSimulator
    return AerSimulator.from_backend(backend, seed_simulator=seed)


# --------------------------------------------------------------------------
# Swap-network circuit on a line
# --------------------------------------------------------------------------
def build_swap_network_circuit(ham: CostHamiltonian, p: int, budget: int, init: str = "dicke",
                               measure: bool = True) -> QuantumCircuit:
    """XY-mixer QAOA where every two-qubit gate acts on neighbouring line positions.

    ``order[k]`` is the asset currently sitting at line position k. One cost
    layer is n rounds of alternating even/odd neighbour blocks, each block
    being RZZ followed by SWAP; after n rounds every pair has met once and
    the order is exactly reversed, so the nearest-neighbour mixer still acts
    on neighbouring assets. Measurement writes position k into classical bit
    order[k], so counts come back in asset order.
    """
    n = ham.n
    beta, gamma = ParameterVector("β", p), ParameterVector("γ", p)
    circuit = QuantumCircuit(n, n) if measure else QuantumCircuit(n)
    order = list(range(n))
    append_initial_state(circuit, n, budget, init)
    for layer in range(p):
        for k in range(n):
            if ham.h[order[k]] != 0.0:
                circuit.rz(2.0 * gamma[layer] * float(ham.h[order[k]]), k)
        met = set()
        for round_ in range(n):
            for k in range(round_ % 2, n - 1, 2):
                a, b = order[k], order[k + 1]
                coupling = float(ham.J[min(a, b), max(a, b)])
                # RZZ and SWAP go in as ONE two-qubit block. A bare SwapGate
                # would be removed by the transpiler's ElidePermutations pass
                # (optimisation levels 2-3), which then re-routes everything
                # and silently undoes the network.
                block = QuantumCircuit(2, name="zz_swap")
                if coupling != 0.0:
                    block.rzz(2.0 * gamma[layer] * coupling, 0, 1)
                block.swap(0, 1)
                circuit.append(block.to_gate(), [k, k + 1])
                order[k], order[k + 1] = b, a
                met.add((min(a, b), max(a, b)))
        assert len(met) == n * (n - 1) // 2, "swap network missed a pair"
        position = {asset: k for k, asset in enumerate(order)}
        for pairs in chain_pairs(n):
            for (a, b) in pairs:
                assert abs(position[a] - position[b]) == 1, "mixer pair is not adjacent"
                circuit.append(XXPlusYYGate(2.0 * beta[layer]), [position[a], position[b]])
    _assert_beta_then_gamma(circuit, p)
    if measure:
        for k in range(n):
            circuit.measure(k, order[k])
    circuit.metadata = {"final_order": order}
    return circuit


# --------------------------------------------------------------------------
# Qubit selection: the best line of n physical qubits
# --------------------------------------------------------------------------
def _edge_errors(backend) -> dict:
    """Two-qubit gate error for each undirected coupling-map edge."""
    target = backend.target
    two_qubit_gate = next(name for name in ("ecr", "cz", "cx") if name in target.operation_names)
    errors = {}
    for (a, b), props in target[two_qubit_gate].items():
        err = props.error if props is not None and props.error is not None else 1e-2
        key = (min(a, b), max(a, b))
        errors[key] = min(errors.get(key, 1.0), err)
    return errors


def best_line(backend, n: int, paths_per_start: int = 400) -> list:
    """A simple path of n physical qubits with low accumulated error.

    Depth-first search from every qubit (heavy-hex has degree <= 3, so the
    search is small), cheapest edge first, scoring each path by the sum of
    -log(1 - error) over its two-qubit edges and its readout errors.
    """
    errors = _edge_errors(backend)
    neighbours = {}
    for (a, b) in errors:
        neighbours.setdefault(a, []).append(b)
        neighbours.setdefault(b, []).append(a)

    def readout(q):
        props = backend.target["measure"].get((q,))
        return props.error if props is not None and props.error is not None else 1e-2

    penalty = lambda e: -np.log(max(1.0 - min(e, 0.5), 1e-9))
    best_path, best_score = None, np.inf
    for start in sorted(neighbours):
        found = 0
        stack = [([start], penalty(readout(start)))]
        while stack and found < paths_per_start:
            path, score = stack.pop()
            if score >= best_score:
                continue
            if len(path) == n:
                best_path, best_score, found = path, score, found + 1
                continue
            options = [q for q in neighbours[path[-1]] if q not in path]
            # push the worst option first so the best one is explored next
            for q in sorted(options, key=lambda q: -errors[(min(path[-1], q), max(path[-1], q))]):
                edge = errors[(min(path[-1], q), max(path[-1], q))]
                stack.append((path + [q], score + penalty(edge) + penalty(readout(q))))
    if best_path is None:
        raise RuntimeError(f"no line of {n} qubits found")
    return best_path


# --------------------------------------------------------------------------
# Transpilation and statistics
# --------------------------------------------------------------------------
def transpile_for(backend, circuit: QuantumCircuit, seed: int = 0, optimization_level: int = 3,
                  initial_layout: list | None = None):
    """Preset pass manager. Pass ``initial_layout=best_line(backend, n)`` for
    circuits that only use neighbouring line positions."""
    pm = generate_preset_pass_manager(optimization_level=optimization_level, backend=backend,
                                      seed_transpiler=seed, initial_layout=initial_layout)
    return pm.run(circuit)


def two_qubit_stats(isa_circuit: QuantumCircuit) -> dict:
    two = lambda inst: inst.operation.num_qubits == 2
    return {
        "two_qubit_gates": int(sum(1 for inst in isa_circuit.data if two(inst))),
        "two_qubit_depth": int(isa_circuit.depth(two)),
        "depth": int(isa_circuit.depth()),
    }


def measured_physical_qubits(isa_circuit: QuantumCircuit) -> list:
    """Physical qubit read into each classical bit, indexed by classical bit."""
    out = [None] * isa_circuit.num_clbits
    for inst in isa_circuit.data:
        if inst.operation.name == "measure":
            out[isa_circuit.find_bit(inst.clbits[0]).index] = isa_circuit.find_bit(inst.qubits[0]).index
    return out


def sample(sampler_backend, isa_circuit: QuantumCircuit, values, shots: int = 4096) -> dict:
    """Run one bound circuit through the SamplerV2 primitive and return counts.

    Works unchanged for an Aer noisy simulator or a live IBM backend.
    """
    from qiskit_ibm_runtime import SamplerV2
    sampler = SamplerV2(mode=sampler_backend)
    job = sampler.run([(isa_circuit, np.asarray(values, float))], shots=shots)
    data = job.result()[0].data
    register = next(iter(data.keys()))
    return getattr(data, register).get_counts()
