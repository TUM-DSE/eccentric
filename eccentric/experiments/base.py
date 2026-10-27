import os
import time
import tracemalloc
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, fields
from typing import Optional

import psutil
import stim
from qiskit_qec.utils import get_stim_circuits

from ..backends import FakeNativeBackend, QubitTracking, get_backend
from ..codes.gross_code import long_range_pairs
from ..codes import get_code, get_max_d, num_rounds, resolve_schedule
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
    schedule: Optional[str] = None

    def describe(self) -> str:
        return ", ".join(
            f"{f.name}={getattr(self, f.name)}"
            for f in fields(self)
            if f.name not in ("experiment", "num_samples") and getattr(self, f.name) is not None
        )


def load_backend(task: Task):
    if task.backend in ("native", "native_lr"):
        return native_backend(task)
    return get_backend(task.backend, task.backend_size)


def native_backend(task: Task):
    """A device whose couplers are exactly the qubit pairs of the code's syndrome circuit; on
    native_lr the long-range couplers of IBM's layout of bivariate bicycle codes are remote gates"""
    if task.distance is None:
        raise Skip("the native device needs a fixed distance")
    qc = get_code(task.code, task.distance, resolve_cycles(task, task.distance), resolve_code_schedule(task)).qc
    pairs = {tuple(qc.find_bit(q).index for q in inst.qubits) for inst in qc.data if len(inst.qubits) == 2}
    remote = long_range_pairs(task.code, pairs) if task.backend == "native_lr" else ()
    return FakeNativeBackend(qc.num_qubits, pairs, remote)


def resolve_distance(task: Task, backend) -> int:
    if task.distance is not None:
        return task.distance
    num_qubits = SINGLE_QPU_SIZES.get(task.backend, backend.coupling_map.size())
    d = get_max_d(task.code, num_qubits)
    if d < 3:
        raise Skip(f"code does not fit on the backend (distance {d})")
    return d


def resolve_code_schedule(task: Task) -> Optional[str]:
    try:
        return resolve_schedule(task.code, task.schedule)
    except ValueError as e:
        raise Skip(str(e))


def resolve_cycles(task: Task, d: int) -> int:
    if task.cycles is None:
        return d
    if task.cycles <= 1:
        raise Skip("cycles must be greater than 1")
    return task.cycles


def with_detector_coordinates(circuit: stim.Circuit, source: stim.Circuit) -> stim.Circuit:
    """Copy the detector coordinates of the code's Stim circuit, which get lost on the way through
    Qiskit, back onto the converted circuit (decoders like Chromobius need them)"""
    coords = [inst.gate_args_copy() for inst in source.flattened() if inst.name == "DETECTOR"]
    if len(coords) != circuit.num_detectors:
        return circuit
    result = stim.Circuit()
    detectors = iter(coords)
    for inst in circuit.flattened():
        if inst.name == "DETECTOR":
            inst = stim.CircuitInstruction("DETECTOR", inst.targets_copy(), next(detectors))
        result.append(inst)
    return result


def noisy_circuit(task: Task, backend, d: int, cycles: int, stage=lambda name: nullcontext()):
    schedule = resolve_code_schedule(task)
    with stage("circuit"):
        code = get_code(task.code, d, cycles, schedule)
    detectors, logicals = code.stim_detectors()
    if task.translating_method:
        code.qc = translate(code.qc, task.translating_method)
    with stage("transpile"):
        layout_method = task.layout_method or getattr(backend, "layout_method", None)
        code.qc = run_transpiler(code.qc, backend, layout_method, task.routing_method)
    qt = QubitTracking(backend, code.qc)
    with stage("repr"):
        circuit = get_stim_circuits(code.qc, detectors=detectors, logicals=logicals)[0][0]
        circuit = with_detector_coordinates(circuit, code.stim_circuit)
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
        "schedule": resolve_code_schedule(task) or "N/A",
    }


def per_round_error_rate(logical_error_rate: float, rounds: int) -> float:
    # Logical error rate per syndrome round, counting an even number of flips as no error:
    # P = (1 - (1 - 2 p)^rounds) / 2, undefined once P reaches 1/2
    if logical_error_rate >= 0.5:
        return float("nan")
    return (1 - (1 - 2 * logical_error_rate) ** (1 / rounds)) / 2


def result_row(task: Task, d: int, cycles: int, logical_error_rate: str, **extra) -> dict:
    rounds = num_rounds(task.code, cycles)
    return {
        **base_row(task, d, cycles),
        "rounds": rounds,
        "logical_error_rate": logical_error_rate,
        "logical_error_rate_per_round": f"{per_round_error_rate(float(logical_error_rate), rounds):.4g}",
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
