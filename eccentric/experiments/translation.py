import numpy as np

from ..codes import get_code
from ..metrics import (
    count_2q_gates_qiskit,
    count_total_gates_qiskit,
    get_resource_overhead_2q_gates,
    get_resource_overhead_total_gates,
)
from ..transpilers import translate
from .base import load_backend, resolve_code_schedule, resolve_distance

REQUIRED = ("backends", "codes", "translating_methods")
DEFAULTS = {"optimization_levels": [0, 2, 3]}


def run(task):
    backend = load_backend(task)
    d = resolve_distance(task, backend)
    schedule = resolve_code_schedule(task)
    code = get_code(task.code, d, task.cycles, schedule)
    level = task.optimization_level
    rows = []
    for state, circuit in code.circuit.items():
        gates, tq_gates = [], []
        for _ in range(task.num_samples):
            translated = translate(circuit, task.translating_method, task.gate_set, level)
            gates.append(get_resource_overhead_total_gates(circuit, translated))
            tq_gates.append(get_resource_overhead_2q_gates(circuit, translated))
        rows.append({
            "backend": task.backend,
            "backend_size": task.backend_size,
            "code": task.code,
            "distance": d,
            "cycles": task.cycles or d,
            "state": state,
            "translating_method": f"{task.translating_method}_{level}",
            "gate_set": task.gate_set or "N/A",
            "schedule": schedule or "N/A",
            "original_total_gates": count_total_gates_qiskit(circuit),
            "original_2q_gates": count_2q_gates_qiskit(circuit),
            "gate_overhead_mean": np.mean(gates),
            "gate_overhead_var": np.var(gates),
            "tq_gate_overhead_mean": np.mean(tq_gates),
            "tq_gate_overhead_var": np.var(tq_gates),
        })
    return rows
