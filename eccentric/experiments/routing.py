import numpy as np

from ..codes import get_code
from ..metrics import count_swaps_qiskit, get_resource_overhead_2q_gates, get_resource_overhead_total_gates
from ..transpilers import run_transpiler
from .base import load_backend, resolve_code_schedule, resolve_distance

REQUIRED = ("backends", "codes")


def run(task):
    backend = load_backend(task)
    d = resolve_distance(task, backend)
    schedule = resolve_code_schedule(task)
    code = get_code(task.code, d, task.cycles, schedule)
    rows = []
    for state, circuit in code.circuit.items():
        gates, tq_gates, swaps = [], [], []
        for _ in range(task.num_samples):
            transpiled = run_transpiler(circuit, backend, task.layout_method, task.routing_method)
            gates.append(get_resource_overhead_total_gates(circuit, transpiled))
            tq_gates.append(get_resource_overhead_2q_gates(circuit, transpiled))
            swaps.append(count_swaps_qiskit(transpiled))
        rows.append({
            "backend": task.backend,
            "backend_size": task.backend_size,
            "code": task.code,
            "distance": d,
            "cycles": task.cycles or d,
            "state": state,
            "routing_method": task.routing_method,
            "layout_method": task.layout_method,
            "schedule": schedule or "N/A",
            "gate_overhead_mean": np.mean(gates),
            "gate_overhead_var": np.var(gates),
            "tq_gate_overhead_mean": np.mean(tq_gates),
            "tq_gate_overhead_var": np.var(tq_gates),
            "swap_overhead_mean": np.mean(swaps),
            "swap_overhead_var": np.var(swaps),
        })
    return rows
