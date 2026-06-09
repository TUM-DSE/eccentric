import sys
import os
import csv
import multiprocessing as mp
from itertools import product
from tqdm import tqdm
import matplotlib.pyplot as plt
import stim
import numpy as np

# Add qec library
sys.path.append(os.path.join(os.getcwd(), "external/qiskit_qec/src"))

from codes import get_code
from backends import get_backend, QubitTracking
from noise import get_noise_model
from decoders import decode

# ==============================================================================
# Constants & Configurations
# ==============================================================================

DISTANCES = [3, 5, 7, 9]
BACKEND_NAME = "real_heron"
DECODING_TIME_NS = 500
NUM_SAMPLES = 10_000

# MCM latencies and error rates from benchmark_mcm.py
MCM_TRADE_OFFS = [
    (200, 0.001), (500, 0.001), (700, 0.001), (1000, 0.001), (1500, 0.001), (2000, 0.001)
]

DEFAULT_MCM_LATENCY = 2000.0
DEFAULT_MCM_ERROR = 0.001

# ==============================================================================
# Simulation Worker
# ==============================================================================

def run_simulation(args):
    code_name, d, num_rounds, latency, error_rate, decoding_time_ns, num_samples = args
    
    # 1. Setup Backend
    backend = get_backend(BACKEND_NAME, None)
    
    # 2. Setup Code
    code = get_code(code_name, d, num_rounds)
    stim_circuit = code.stim_circuit
    
    # Qubit tracking
    qt = QubitTracking(backend, getattr(code, 'qc', None))
    
    # 3. Setup Noise Model
    m_time_multiplier = latency / DEFAULT_MCM_LATENCY
    m_error_multiplier = error_rate / DEFAULT_MCM_ERROR
    error_type = f"{BACKEND_NAME}_{m_error_multiplier}_{m_time_multiplier}"
    
    idle_multiplier = 3.0 # Heron uses 3.0
    
    noise_model = get_noise_model(error_type, qt, None, backend, idle_multiplier=idle_multiplier)
    noisy_stim_circuit = noise_model.noisy_circuit(stim_circuit)
    
    # 3.5 Inject Decoder Backlog Reaction Idle Time
    t_mcm_s = latency * 1e-9
    t_gates_s = 300 * 1e-9
    t_cycle_s = t_gates_s + t_mcm_s
    
    decoding_time_s = decoding_time_ns * 1e-9
    t_budget_s = 1000 * 1e-9
    
    t_excess_quantum = max(0.0, t_cycle_s - t_budget_s)
    t_excess_decoder = max(0.0, decoding_time_s - t_budget_s)
    
    t_idle_s = num_rounds * (t_excess_quantum + t_excess_decoder)
    
    if t_idle_s > 0:
        final_m_start = len(noisy_stim_circuit)
        for i in range(len(noisy_stim_circuit)-1, -1, -1):
            op = noisy_stim_circuit[i]
            if isinstance(op, stim.CircuitRepeatBlock):
                final_m_start = i + 1
                if final_m_start < len(noisy_stim_circuit) and noisy_stim_circuit[final_m_start].name == 'TICK':
                    final_m_start += 1
                break
        
        prefix = noisy_stim_circuit[0:final_m_start]
        suffix = noisy_stim_circuit[final_m_start:]
        delay_circuit = noise_model.get_idle_delay_circuit(t_idle_s, noisy_stim_circuit.num_qubits)
        noisy_stim_circuit = prefix + delay_circuit + suffix
    
    # 4. Decode
    decoder_name = "mwpm"
    try:
        ler = decode(code_name, noisy_stim_circuit, num_samples, decoder_name, BACKEND_NAME, error_type)
    except Exception as e:
        print(f"Error decoding {code_name} d={d} rounds={num_rounds} latency={latency}: {e}")
        ler = None
        
    return (d, num_rounds, latency, ler)

# ==============================================================================
# Main
# ==============================================================================

def main():
    print(f"Starting Rounds-LER Benchmark (Samples: {NUM_SAMPLES}, Distances: {DISTANCES})...")
    
    os.makedirs("experiment_results/benchmark", exist_ok=True)
    csv_path = "experiment_results/benchmark/rounds_ler_benchmark.csv"
    
    tasks = []
    for d in DISTANCES:
        # Non-linear rounds: d, 2d, 5d, 100, 200, 500, 1000 (filtered for > d and unique)
        rounds_to_test = sorted(list(set([d, 2*d, 5*d, 100, 200, 500])))
        rounds_to_test = [r for r in rounds_to_test if r >= d and r <= 500]
        
        for num_rounds in rounds_to_test:
            for latency, error_rate in MCM_TRADE_OFFS:
                tasks.append(("surface", d, num_rounds, latency, error_rate, DECODING_TIME_NS, NUM_SAMPLES))
                    
    print(f"Total configurations to evaluate: {len(tasks)}")
    
    results = []
    num_procs = max(1, mp.cpu_count() - 2)
    with mp.Pool(num_procs) as pool:
        for res in tqdm(pool.imap_unordered(run_simulation, tasks), total=len(tasks)):
            if res[-1] is not None:
                results.append(res)
                
    # Save CSV
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["distance", "rounds", "mcm_latency_ns", "ler"])
        writer.writerows(results)
    
    print(f"\\nSaved raw results to {csv_path}")
    
    # ==========================================================================
    # Plotting
    # ==========================================================================
    for d in DISTANCES:
        plt.figure(figsize=(10, 6))
        
        # Filter for this distance
        dist_res = [r for r in results if r[0] == d]
        if not dist_res:
            plt.close()
            continue
            
        latencies = sorted(list(set(r[2] for r in dist_res)))
        
        for latency in latencies:
            branch = [r for r in dist_res if r[2] == latency]
            branch.sort(key=lambda x: x[1]) # sort by rounds
            
            if branch:
                x = [r[1] for r in branch]
                y = [r[3] for r in branch]
                plt.plot(x, y, marker='o', label=f"MCM: {latency}ns")

        plt.xlabel("Number of Rounds")
        plt.ylabel("Logical Error Rate (LER)")
        plt.title(f"LER vs Rounds (Surface Code d={d}, Decode: {DECODING_TIME_NS}ns)")
        plt.yscale("log")
        plt.xscale("log")
        plt.grid(True, which="both", ls="--", alpha=0.5)
        plt.legend()
        
        plot_path = f"experiment_results/benchmark/plot_rounds_ler_d{d}.pdf"
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close()
        print(f"Saved plot: {plot_path}")

if __name__ == "__main__":
    main()
