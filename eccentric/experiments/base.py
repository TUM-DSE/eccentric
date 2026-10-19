import os
import time
import tracemalloc
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, fields
from typing import Optional

import psutil
from qiskit_qec.utils import get_stim_circuits

from ..backends import QubitTracking, get_backend
from ..codes import get_code, get_max_d
from ..noise import get_noise_model
from ..transpilers import run_transpiler, translate

SINGLE_QPU_SIZES = {
    "real_flamingo_1_qpu": 133,
    "real_loon_1_qpu": 120,
}


class Skip(Exception):
    pass


@dataclass(frozen=True)
class Task:
    experiment: str
    num_samples: int
    backend: str
    code: str
    backend_size: Optional[int] = None
    distance: Optional[int] = None
    cycles: Optional[int] = None
    decoder: Optional[str] = None
    error_type: Optional[str] = None
    error_probability: Optional[float] = None
    layout_method: Optional[str] = None
    routing_method: Optional[str] = None
    translating_method: Optional[str] = None
    gate_set: Optional[str] = None
    optimization_level: Optional[int] = None

    def describe(self) -> str:
        return ", ".join(
            f"{f.name}={getattr(self, f.name)}"
            for f in fields(self)
            if f.name not in ("experiment", "num_samples") and getattr(self, f.name) is not None
        )


def load_backend(task: Task):
    return get_backend(task.backend, task.backend_size)


def resolve_distance(task: Task, backend) -> int:
    if task.distance is not None:
        return task.distance
    num_qubits = SINGLE_QPU_SIZES.get(task.backend, backend.coupling_map.size())
    d = get_max_d(task.code, num_qubits)
    if d < 3:
        raise Skip(f"code does not fit on the backend (distance {d})")
    return d


def resolve_cycles(task: Task, d: int) -> int:
    if task.cycles is None:
        return d
    if task.cycles <= 1:
        raise Skip("cycles must be greater than 1")
    return task.cycles


def noisy_circuit(task: Task, backend, d: int, cycles: int, stage=lambda name: nullcontext()):
    with stage("circuit"):
        code = get_code(task.code, d, cycles)
    detectors, logicals = code.stim_detectors()
    if task.translating_method:
        code.qc = translate(code.qc, task.translating_method)
    with stage("transpile"):
        code.qc = run_transpiler(code.qc, backend, task.layout_method, task.routing_method)
    qt = QubitTracking(backend, code.qc)
    with stage("repr"):
        circuit = get_stim_circuits(code.qc, detectors=detectors, logicals=logicals)[0][0]
    with stage("noise"):
        noise_model = get_noise_model(task.error_type, qt, task.error_probability, backend)
        circuit = noise_model.noisy_circuit(circuit)
    return circuit


def base_row(task: Task, d: int, cycles: int) -> dict:
    return {
        "backend": task.backend,
        "backend_size": task.backend_size,
        "code": task.code,
        "decoder": task.decoder,
        "distance": d,
        "cycles": cycles,
        "num_samples": task.num_samples,
        "error_type": task.error_type,
        "error_probability": task.error_probability,
    }


def result_row(task: Task, d: int, cycles: int, logical_error_rate: str, **extra) -> dict:
    return {
        **base_row(task, d, cycles),
        "logical_error_rate": logical_error_rate,
        "layout_method": task.layout_method or "N/A",
        "routing_method": task.routing_method or "N/A",
        "translating_method": task.translating_method or "N/A",
        **extra,
    }


class Profiler:
    def __init__(self):
        self.process = psutil.Process(os.getpid())
        self.times = {}
        self.heap = {}
        self.rss = {}
        tracemalloc.start()
        self.start = time.perf_counter()

    @contextmanager
    def stage(self, name: str):
        t0 = time.perf_counter()
        yield
        self.times[name] = time.perf_counter() - t0
        self.heap[name] = tracemalloc.get_traced_memory()[1] / 1e6
        self.rss[name] = self.process.memory_info().rss / 1e6

    def columns(self, totals: bool = False) -> dict:
        row = {f"time_{name}": value for name, value in self.times.items()}
        if totals:
            row["total_time"] = time.perf_counter() - self.start
        row.update({f"mem_{name}_MB": value for name, value in self.heap.items()})
        row.update({f"mem_{name}_full_MB": value for name, value in self.rss.items()})
        if totals:
            row["peak_memory_MB"] = tracemalloc.get_traced_memory()[1] / 1e6
            row["peak_memory_full_MB"] = self.process.memory_info().rss / 1e6
        tracemalloc.stop()
        return row
