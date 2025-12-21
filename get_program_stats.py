import sys
import time
import psutil
import tracemalloc
import os
import stim

sys.path.append(os.path.join(os.getcwd(), "external/qiskit_qec/src"))

import yaml
import logging

from itertools import product
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Manager
from qiskit.compiler import transpile
from qiskit_qec.utils import get_stim_circuits
from backends import get_backend, QubitTracking
from codes import get_code, get_max_d, get_min_n
from noise import get_noise_model
from decoders import decode
from transpilers import run_transpiler, translate
from utils import save_experiment_metadata, save_results_to_csv, setup_experiment_logging
import stim

def run_experiment(
    experiment_name,
    backend_name,
    backend_size,
    code_name,
    decoder,
    d,
    cycles,
    num_samples,
    error_type,
    error_prob,
    lock,
    layout_method=None,
    routing_method=None,
    translating_method=None,
):
    try:
        pid = os.getpid()
        process = psutil.Process(pid)
        # Start total timing and memory tracking
        total_start = time.perf_counter()
        tracemalloc.start()

        # -----------------------
        # BACKEND SETUP
        t0 = time.perf_counter()
        backend = get_backend(backend_name, backend_size)
        t_backend = time.perf_counter() - t0
        _, mem_backend = tracemalloc.get_traced_memory()
        mem_backend_full = process.memory_info().rss / 1e6

        if d is None:
            if backend_name == "real_flamingo_1_qpu":
                d = get_max_d(code_name, 133)
            elif backend_name == "real_loon_1_qpu":
                d = get_max_d(code_name, 120)
            else:
                d = get_max_d(code_name, backend.coupling_map.size())
            if d < 3:
                logging.info(f"{experiment_name} | Execution not possible: distance {d}")
                return

        if cycles is None:
            cycles = d

        # -----------------------
        # CIRCUIT GENERATION
        t0 = time.perf_counter()
        code = get_code(code_name, d, cycles)
        t_circuit = time.perf_counter() - t0
        _, mem_circuit = tracemalloc.get_traced_memory()
        mem_circuit_full = process.memory_info().rss / 1e6

        detectors, logicals = code.stim_detectors()
        if translating_method:
            code.qc = translate(code.qc, translating_method)

        # -----------------------
        # TRANSPILATION
        t0 = time.perf_counter()
        code.qc = run_transpiler(code.qc, backend, layout_method, routing_method)
        t_transpile = time.perf_counter() - t0
        _, mem_transpile = tracemalloc.get_traced_memory()
        mem_transpile_full = process.memory_info().rss / 1e6
        qt = QubitTracking(backend, code.qc)

        # -----------------------
        # REPRESENTATION CHANGE
        t0 = time.perf_counter()
        stim_circuit = get_stim_circuits(code.qc, detectors=detectors, logicals=logicals)[0][0]
        t_repr = time.perf_counter() - t0
        _, mem_repr = tracemalloc.get_traced_memory()
        mem_repr_full = process.memory_info().rss / 1e6

        # -----------------------
        # NOISE ADDITION
        t0 = time.perf_counter()
        noise_model = get_noise_model(error_type, qt, error_prob, backend)
        stim_circuit = noise_model.noisy_circuit(stim_circuit)
        t_noise = time.perf_counter() - t0
        _, mem_noise = tracemalloc.get_traced_memory()
        mem_noise_full = process.memory_info().rss / 1e6

        # -----------------------
        # DECODING
        t0 = time.perf_counter()
        logical_error_rate = decode(code_name, stim_circuit, num_samples, decoder, backend_name, error_type)
        t_decode = time.perf_counter() - t0
        _, mem_decode = tracemalloc.get_traced_memory()
        mem_decode_full = process.memory_info().rss / 1e6

        # Total elapsed time and peak memory
        total_elapsed = time.perf_counter() - total_start
        _, peak_mem = tracemalloc.get_traced_memory()
        peak_mem_full = process.memory_info().rss / 1e6
        tracemalloc.stop()

        # Save results
        result_data = {
            "backend": backend_name,
            "backend_size": backend_size,
            "code": code_name,
            "decoder": decoder,
            "distance": d,
            "cycles": cycles,
            "num_samples": num_samples,
            "error_type": error_type,
            "error_probability": error_prob,
            "logical_error_rate": f"{logical_error_rate:.6f}",
            "layout_method": layout_method or "N/A",
            "routing_method": routing_method or "N/A",
            "translating_method": translating_method or "N/A",
            "time_backend": t_backend,
            "time_circuit": t_circuit,
            "time_transpile": t_transpile,
            "time_repr": t_repr,
            "time_noise": t_noise,
            "time_decode": t_decode,
            "total_time": total_elapsed,
            "mem_backend_MB": mem_backend / 1e6,
            "mem_circuit_MB": mem_circuit / 1e6,
            "mem_transpile_MB": mem_transpile / 1e6,
            "mem_repr_MB": mem_repr / 1e6,
            "mem_noise_MB": mem_noise / 1e6,
            "mem_decode_MB": mem_decode / 1e6,
            "mem_backend_full_MB": mem_backend_full,
            "mem_circuit_full_MB": mem_circuit_full,
            "mem_transpile_full_MB": mem_transpile_full,
            "mem_repr_full_MB": mem_repr_full,
            "mem_noise_full_MB": mem_noise_full,
            "mem_decode_full_MB": mem_decode_full,
            "peak_memory_MB": peak_mem / 1e6,
            "peak_memory_full_MB": peak_mem_full
        }

        with lock:
            save_results_to_csv(result_data, experiment_name)

        print(f"[PID {pid}] Completed experiment: total_time={total_elapsed:.2f}s, "
              f"peak_mem_heap={peak_mem/1e6:.2f} MB, peak_mem_full={peak_mem_full:.2f} MB")

    except Exception as e:
        logging.error(f"{experiment_name} | Failed experiment: {e}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Remember to add YAML file!")
        sys.exit(1)
    
    conf_file = sys.argv[1]

    with open(conf_file, "r") as f:
        config = yaml.safe_load(f)

    for experiment in config["experiments"]:
        experiment_name = experiment["name"]
        num_samples = experiment["num_samples"]
        backends = experiment["backends"]
        codes = experiment["codes"]
        decoders = experiment["decoders"]
        error_types = experiment["error_types"]
        error_probabilities = experiment.get("error_probabilities", [None])
        cycles = experiment.get("cycles", None)
        layout_methods = experiment.get("layout_methods", [None])
        routing_methods = experiment.get("routing_methods", [None])
        translating_methods = experiment.get("translating_methods", [None])

        setup_experiment_logging(experiment_name)
        save_experiment_metadata(experiment, experiment_name)
        manager = Manager()
        lock = manager.Lock()
        # TODO: better handling case if distances and backends_sizes are both set

        with ProcessPoolExecutor() as executor:
            if "backends_sizes" in experiment and "distances" in experiment:
                raise ValueError("Cannot set both backends_sizes and distances in the same experiment")
            if "distances" in experiment:
                distances = experiment["distances"]
                parameter_combinations = product(backends, codes, decoders, error_types, error_probabilities, distances, layout_methods, routing_methods, translating_methods)
                futures = [
                    executor.submit(
                        run_experiment,
                        experiment_name,
                        backend,
                        get_min_n(code_name, d),
                        code_name,
                        decoder,
                        d,
                        cycles,
                        num_samples,
                        error_type,
                        error_prob,
                        lock,
                        layout_method,
                        routing_method,
                        translating_method
                    )
                    for backend, code_name, decoder, error_type, error_prob, d, layout_method, routing_method, translating_method in parameter_combinations
                ]
            elif "backends_sizes" in experiment:
                backends_sizes = experiment["backends_sizes"]
                parameter_combinations = product(
                    backends, backends_sizes, codes, decoders, error_types, error_probabilities, layout_methods, routing_methods, translating_methods
                )
                futures = [
                    executor.submit(
                        run_experiment,
                        experiment_name,
                        backend,
                        backends_sizes,
                        code_name,
                        decoder,
                        None,
                        cycles,
                        num_samples,
                        error_type,
                        error_prob,
                        lock,
                        layout_method,
                        routing_method,
                        translating_method,
                    )
                    for backend, backends_sizes, code_name, decoder, error_type, error_prob, layout_method, routing_method, translating_method in parameter_combinations
                ]
            else:
                parameter_combinations = product(backends, codes, decoders, error_types, error_probabilities, layout_methods, routing_methods, translating_methods)
                futures = [
                    executor.submit(
                        run_experiment,
                        experiment_name,
                        backend,
                        None,
                        code_name,
                        decoder,
                        None,
                        cycles,
                        num_samples,
                        error_type,
                        error_prob,
                        lock,
                        layout_method,
                        routing_method,
                        translating_method,
                    )
                    for backend, code_name, decoder, error_type, error_prob, layout_method, routing_method, translating_method in parameter_combinations
                ]
            for future in futures:
                future.result()
