import sys
import os
import csv
import multiprocessing as mp
from itertools import product
from tqdm import tqdm
import matplotlib.pyplot as plt
import stim

# Add qec library
sys.path.append(os.path.join(os.getcwd(), "external/qiskit_qec/src"))

from codes import get_code
from backends import get_backend, QubitTracking
from noise import get_noise_model
from decoders import decode

# ==============================================================================
# Constants & Configurations
# ==============================================================================

CODES = {
    "surface": [3, 5, 7, 9, 11],
    "color": [3, 5, 7, 9, 11],
    "steane": [3, 9],
    #"gross": [12]
}

#BACKENDS = ["real_flamingo", "real_heron"]
BACKENDS = ["real_heron"]

# Tradeoff points extracted from the plot: method -> [(latency_ns, error_rate)]
# Error rate is computed as 1 - Geometric Mean Fidelity
MCM_METHODS = {
    "Mamba": [
        (200, 0.001), (500, 0.001), (700, 0.001), (1000, 0.001), (1500, 0.001), (2000, 0.001)
    ],
}
DEFAULT_MCM_LATENCY = 2000.0
DEFAULT_MCM_ERROR = 0.001

DECODING_TIMES = [500, 1000, 2000, 5000, 10000] # Decoder reaction times (ns)

NUM_SAMPLES = 20_000

# ==============================================================================
# Simulation Worker
# ==============================================================================

def run_simulation(args):
    code_name, expected_d, backend_name, method_name, latency, error_rate, decoding_time_ns, num_samples = args
    
    # 1. Setup Backend
    backend = get_backend(backend_name, None)
    
    # 2. Setup Code
    cycles = expected_d
    code = get_code(code_name, expected_d, cycles)
    stim_circuit = code.stim_circuit
    
    # Qubit tracking (trivial layout for code.qc = None)
    qt = QubitTracking(backend, getattr(code, 'qc', None))
    
    # 3. Setup Noise Model
    # Both Flamingo and Heron interpret these multipliers relative to their defaults
    m_time_multiplier = latency / DEFAULT_MCM_LATENCY
    m_error_multiplier = error_rate / DEFAULT_MCM_ERROR
    error_type = f"{backend_name}_{m_error_multiplier}_{m_time_multiplier}"
    
    # IBM backends (Heron, Flamingo) use a multiplier of 3 to account for extra crosstalk/thermal noise
    idle_multiplier = 3.0 if ("heron" in backend_name.lower() or "flamingo" in backend_name.lower()) else 1.0
    
    noise_model = get_noise_model(error_type, qt, None, backend, idle_multiplier=idle_multiplier)
    noisy_stim_circuit = noise_model.noisy_circuit(stim_circuit)
    
    # 3.5 Inject Decoder Backlog Reaction Idle Time
    # Real physical time of the MCM readout
    t_mcm_s = latency * 1e-9
    t_gates_s = 300 * 1e-9  # Approx duration of logic gates in one cycle
    t_cycle_s = t_gates_s + t_mcm_s
    
    decoding_time_s = decoding_time_ns * 1e-9
    
    # Fixed budget of 1us
    t_budget_s = 1000 * 1e-9
    
    # Backlog occurs if the quantum cycle or the decoder exceeds the 1us budget
    t_excess_quantum = max(0.0, t_cycle_s - t_budget_s)
    t_excess_decoder = max(0.0, decoding_time_s - t_budget_s)
    
    # Total idle time from both potential sources of backlog
    t_idle_s = expected_d * (t_excess_quantum + t_excess_decoder)
    
    if t_idle_s > 0:
        final_m_start = len(noisy_stim_circuit)
        # We split the circuit right after the last REPEAT block and any trailing TICKs.
        # This guarantees we inject the delay before any pre-measurement noise associated with the final M block.
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
    # bposd works for color, steane, gross, surface
    decoder_name = "mwpm"
    
    try:
        ler = decode(code_name, noisy_stim_circuit, num_samples, decoder_name, backend_name, error_type)
    except Exception as e:
        print(f"Error decoding {code_name} d={expected_d} on {backend_name}: {e}")
        ler = None
        
    return (code_name, expected_d, backend_name, method_name, latency, error_rate, decoding_time_ns, ler)

# ==============================================================================
# Main
# ==============================================================================

def main():
    print("Starting MCM Latency-Fidelity Benchmark...")
    
    tasks = []
    for code_name, distances in CODES.items():
        for d in distances:
            for backend_name in BACKENDS:
                for method_name, tradeoffs in MCM_METHODS.items():
                    for latency, error_rate in tradeoffs:
                        for dec_time in DECODING_TIMES:
                            tasks.append((code_name, d, backend_name, method_name, latency, error_rate, dec_time, NUM_SAMPLES))
                    
    print(f"Total configurations to evaluate: {len(tasks)}")
    
    results = []
    # Use fewer processes to not run out of memory with BPOSD
    num_procs = max(1, mp.cpu_count() - 2)
    with mp.Pool(num_procs) as pool:
        for res in tqdm(pool.imap_unordered(run_simulation, tasks), total=len(tasks)):
            if res[-1] is not None:
                results.append(res)
                
    # Save CSV
    os.makedirs("experiment_results/benchmark", exist_ok=True)
    csv_path = "experiment_results/benchmark/mcm_latency_benchmark.csv"
    
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["code", "distance", "backend", "method", "readout_duration_ns", "readout_error", "decoding_time_ns", "ler"])
        writer.writerows(results)
    
    print(f"\\nSaved raw results to {csv_path}")
    
    # ==========================================================================
    # Plotting
    # ==========================================================================
    # Group results by code
    # results format: code_name, d, backend_name, method_name, latency, error_rate, ler
    
    colors = {3: 'tab:blue', 5: 'tab:orange', 7: 'tab:green', 9: 'tab:red', 11: 'tab:purple', 12: 'tab:brown'}
    markers = {3: 'o', 5: 's', 7: '^', 9: 'D', 11: 'v', 12: 'p'}
    lines = {"real_flamingo": "-", "real_heron": "--"}
    
    for dec_time in DECODING_TIMES:
        for code_name in CODES.keys():
            plt.figure(figsize=(10, 6))
            
            # Filter for this code and dec_time
            code_res = [r for r in results if r[0] == code_name and r[6] == dec_time]
            if not code_res:
                plt.close()
                continue
                
            distances = sorted(list(set(r[1] for r in code_res)))
            
            for d in distances:
                for method in MCM_METHODS.keys():
                    for backend, linestyle in lines.items():
                        branch = [r for r in code_res if r[1] == d and r[2] == backend and r[3] == method]
                        branch.sort(key=lambda x: x[4]) # sort by readout latency
                        
                        if branch:
                            x = [r[4] for r in branch]
                            y = [r[7] for r in branch]
                            
                            label = f"{method} d={d} ({'Flamingo' if 'flamingo' in backend else 'Heron'})"
                            plt.plot(x, y, marker=markers.get(d, 'X'), linestyle=linestyle, color=colors.get(d, 'black'), label=label)

            plt.xlabel("Readout Duration (ns)")
            plt.ylabel("Logical Error Rate (LER)")
            plt.title(f"MCM Latency-Fidelity Tradeoff: {code_name.capitalize()} Code (Decode: {dec_time}ns)")
            plt.yscale("log")
            plt.grid(True, which="both", ls="--", alpha=0.5)
            # Place legend outside to avoid obscuring data
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            
            plot_path = f"experiment_results/benchmark/plot_{code_name}_dec{dec_time}ns.pdf"
            plt.tight_layout()
            plt.savefig(plot_path)
            plt.close()
            print(f"Saved plot: {plot_path}")

if __name__ == "__main__":
    main()
