import numpy as np
import random
import os
import json
from qiskit import QuantumCircuit, transpile
from qiskit_ibm_runtime.fake_provider import FakeTorino
from qiskit_aer import AerSimulator
from typing import Dict, List, Any, Optional, Tuple
from noise import get_noise_model
from noise.heron_noise import HeronNoise
from noise.artificial_noise import ArtificialNoise
from backends import QubitTracking
from qiskit_aer.noise import NoiseModel, depolarizing_error, ReadoutError
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime



def load_results_from_file(job_id: str) -> Dict[str, Any]:
    results_file = f"job_results_{job_id}.json"
    if os.path.exists(results_file):
        print(f"Loading results from local file: {results_file}")
        with open(results_file, 'r') as f:
            return json.load(f)
    return None


def load_job_info_from_file(job_id: str) -> Dict[str, Any]:
    job_file = f"job_info_{job_id}.json"
    if os.path.exists(job_file):
        with open(job_file, 'r') as f:
            return json.load(f)
    else:
        print(f"Job info file not found: {job_file}")
        return {}


def _normalize_counts_format(counts: Dict[str, int]) -> Dict[str, int]:
    sample_key = list(counts.keys())[0]
    if ' ' in sample_key:
        normalized = {}
        for bitstring, count in counts.items():
            parts = bitstring.split()
            
            active_register = None
            for part in parts:
                if part != '0' * len(part):
                    active_register = part
                    break
            if active_register is None:
                active_register = parts[0]
            normalized[active_register] = normalized.get(active_register, 0) + count
        return normalized
    else:
        return counts

def _calculate_ground_state_fidelity(counts: Dict[str, int]) -> float:
    normalized_counts = _normalize_counts_format(counts)
    total_shots = sum(normalized_counts.values())
    if total_shots == 0:
        return 0.0
    
    if normalized_counts:
        sample_key = list(normalized_counts.keys())[0]
        ground_pattern = '0' * len(sample_key)
    else:
        return 0.0
    
    ground_count = normalized_counts.get(ground_pattern, 0)
    return ground_count / total_shots

def _recreate_experimental_circuits(
    qubit_range: Tuple[int, int] = (2, 7),
    circuits_per_qubit_size: int = 10,
    circuit_depth: int = 100,
    seed: int = 42
) -> Tuple[List[QuantumCircuit], List[Dict]]:

    random.seed(seed)
    np.random.seed(seed)
    
    qubit_sizes = list(range(qubit_range[0], qubit_range[1] + 1))
    circuits = []
    circuit_metadata = []
    
    for qubit_size in qubit_sizes:
        for i in range(circuits_per_qubit_size):
            circuit_seed = seed + qubit_size * 1000 + i
            
            circuit = generate_random_circuit_with_torino_gates(
                num_qubits=qubit_size,
                depth=circuit_depth,
                seed=circuit_seed
            )
            
            circuits.append(circuit)
            circuit_metadata.append({
                'qubit_size': qubit_size,
                'circuit_index': i,
                'depth': circuit.depth(),
                'size': circuit.size()
            })
    
    return circuits, circuit_metadata

def generate_random_circuit_with_torino_gates(
    num_qubits: int,
    depth: int,
    seed: Optional[int] = None
) -> QuantumCircuit:

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    

    single_qubit_gates = ['x', 'sx', 'rz', 'id'] 
    
    circuit = QuantumCircuit(num_qubits, num_qubits)
    
    for layer in range(depth):
        available_qubits = list(range(num_qubits))
        random.shuffle(available_qubits)
        
        used_qubits = set()
        num_two_qubit = random.randint(0, min(2, num_qubits // 2))
        for _ in range(num_two_qubit):
            if len(available_qubits) < 2:
                break

            available_pairs = [(i, j) for i in available_qubits 
                             for j in available_qubits 
                             if i != j and i not in used_qubits and j not in used_qubits]
            
            if not available_pairs:
                break
                
            qubit1, qubit2 = random.choice(available_pairs)
            circuit.cz(qubit1, qubit2)
            used_qubits.update([qubit1, qubit2])
        
        remaining_qubits = [q for q in available_qubits if q not in used_qubits]
        for qubit in remaining_qubits:
            if random.random() < 0.7:  # 70% chance to apply a gate
                gate_type = random.choice(single_qubit_gates)
                
                if gate_type == 'x':
                    circuit.x(qubit)
                elif gate_type == 'sx':
                    circuit.sx(qubit)
                elif gate_type == 'rz':
                    angle = random.uniform(0, 2 * np.pi)
                    circuit.rz(angle, qubit)
                elif gate_type == 'id':
                    circuit.id(qubit)
        
        if layer < depth - 1:
            circuit.barrier()
    circuit.measure_all()
    
    return circuit

def _calculate_fidelity(ideal_counts: Dict[str, int], noisy_counts: Dict[str, int]) -> float:
    ideal_normalized = _normalize_counts_format(ideal_counts)
    noisy_normalized = _normalize_counts_format(noisy_counts)
    
    total_ideal = sum(ideal_normalized.values())
    total_noisy = sum(noisy_normalized.values())
    
    if total_ideal == 0 or total_noisy == 0:
        return 0.0
    
    all_outcomes = set(ideal_normalized.keys()) | set(noisy_normalized.keys())
    
    tv_distance = 0.0
    for outcome in all_outcomes:
        p_ideal = ideal_normalized.get(outcome, 0) / total_ideal
        p_noisy = noisy_normalized.get(outcome, 0) / total_noisy
        tv_distance += abs(p_ideal - p_noisy)
    
    fidelity = 1.0 - 0.5 * tv_distance
    return max(0.0, fidelity)

def _create_qiskit_noise_approximation(custom_noise_model, backend, use_cz=False):

    noise_model = NoiseModel()
    if hasattr(custom_noise_model, 'sq') and custom_noise_model.sq > 0:
        sq_error = depolarizing_error(custom_noise_model.sq, 1)
        noise_model.add_all_qubit_quantum_error(sq_error, ['sx', 'x', 'rz'])
    
    if hasattr(custom_noise_model, 'tq') and custom_noise_model.tq > 0:
        tq_error = depolarizing_error(custom_noise_model.tq, 2)
        if use_cz:
            noise_model.add_all_qubit_quantum_error(tq_error, ['cz'])
        else:
            noise_model.add_all_qubit_quantum_error(tq_error, ['cx', 'cz'])
    
    if hasattr(custom_noise_model, 'measure') and custom_noise_model.measure > 0:
        measure_error = ReadoutError([[1-custom_noise_model.measure, custom_noise_model.measure],
                                    [custom_noise_model.measure, 1-custom_noise_model.measure]])
        noise_model.add_all_qubit_readout_error(measure_error)

    if hasattr(custom_noise_model, 'reset') and custom_noise_model.reset > 0:
        reset_error = depolarizing_error(custom_noise_model.reset, 1)
        noise_model.add_all_qubit_quantum_error(reset_error, ['reset'])
    
    return noise_model

def _generate_comparison_summary_ibm(results: Dict[str, Any]) -> Dict[str, Any]:
    summary = {
        'fidelities': {},
        'best_performing': None,
        'worst_performing': None,
        'analysis': {}
    }
    
    for backend_key in ['noiseless', 'ibm_hardware', 'custom_noise']:
        if backend_key in results:
            summary['fidelities'][backend_key] = results[backend_key]['fidelity']
    
    noisy_fidelities = {k: v for k, v in summary['fidelities'].items() if k != 'noiseless'}
    if noisy_fidelities:
        summary['best_performing'] = max(noisy_fidelities.keys(), key=lambda k: noisy_fidelities[k])
        summary['worst_performing'] = min(noisy_fidelities.keys(), key=lambda k: noisy_fidelities[k])
    
    if 'ibm_hardware' in summary['fidelities'] and 'custom_noise' in summary['fidelities']:
        ibm_fid = summary['fidelities']['ibm_hardware']
        custom_fid = summary['fidelities']['custom_noise']
        diff = abs(ibm_fid - custom_fid)
        
        summary['analysis'] = {
            'fidelity_difference': diff,
            'relative_error': diff / max(ibm_fid, custom_fid) if max(ibm_fid, custom_fid) > 0 else 0,
            'custom_vs_ibm': 'better' if custom_fid > ibm_fid else 'worse' if custom_fid < ibm_fid else 'equal'
        }
    
    return summary

def compare_circuit_execution_with_ibm_cloud(
    circuit: QuantumCircuit,
    ibm_job_results: Dict[str, int],
    circuit_index: int = 0,
    shots: int = 1024,
    noise_types: List[str] = ["heron"],
    noise_param: float = 0.001,
    custom_backend=None,
    custom_noise_model=None,
    ibm_backend_name: str = "ibm_torino"
) -> Dict[str, Any]:
    
    results = {}
    
    if not any(instr.operation.name == 'measure' for instr in circuit.data):
        circuit_with_measurements = circuit.copy()
        if not circuit_with_measurements.cregs:
            circuit_with_measurements.add_register('c', circuit.num_qubits)
        circuit_with_measurements.measure_all()
        circuit = circuit_with_measurements
    
    fake_torino = FakeTorino()
    
    common_transpiled = transpile(
        circuit,
        fake_torino,
        basis_gates=['sx', 'x', 'rz', 'cz', 'id'],
        optimization_level=1
    )
    
    noiseless_simulator = AerSimulator()
    noiseless_job = noiseless_simulator.run(common_transpiled, shots=shots)
    noiseless_result = noiseless_job.result()
    
    results['noiseless'] = {
        'backend': 'AerSimulator (noiseless)',
        'counts': noiseless_result.get_counts(),
        'fidelity': 1.0,
        'execution_time': getattr(noiseless_result, 'time_taken', 'N/A')
    }

    ibm_fidelity = _calculate_fidelity(
        results['noiseless']['counts'],
        ibm_job_results
    )
    
    results['ibm_hardware'] = {
        'backend': f'{ibm_backend_name} (IBM Cloud)',
        'counts': ibm_job_results,
        'fidelity': ibm_fidelity,
        'execution_time': 'N/A', 
        'circuit_index': circuit_index
    }

    if custom_backend is None:
        custom_backend = fake_torino
    
    qt = QubitTracking(custom_backend, common_transpiled)

    for noise_type in noise_types:
        if ':' in noise_type:
            base_type, p_str = noise_type.split(':')
            p_value = float(p_str)
        else:
            base_type = noise_type
            p_value = noise_param

        if base_type == "heron":
            noise_model = HeronNoise.get_noise(qt, custom_backend)
        elif base_type == "modsi1000":
            noise_model = ArtificialNoise.modSI1000(p_value, qt)
        elif base_type == "pc3":
            noise_model = ArtificialNoise.PC3(p_value, qt)
        else:
            noise_model = get_noise_model(base_type, qt, p_value, custom_backend)
        
        if custom_noise_model is not None:
            noise_model = custom_noise_model

        qiskit_custom_noise = _create_qiskit_noise_approximation(noise_model, custom_backend, use_cz=True)
        custom_simulator = AerSimulator(noise_model=qiskit_custom_noise)
        
        custom_job = custom_simulator.run(common_transpiled, shots=shots)
        custom_result = custom_job.result()

        custom_fidelity = _calculate_fidelity(
            results['noiseless']['counts'],
            custom_result.get_counts()
        )

        results[f'{noise_type}_noise'] = {
            'backend': f'Custom backend with {noise_type} noise',
            'counts': custom_result.get_counts(),
            'fidelity': custom_fidelity,
            'execution_time': getattr(custom_result, 'time_taken', 'N/A'),
            'noise_parameters': {
                'sq_error': getattr(noise_model, 'sq', 'N/A'),
                'tq_error': getattr(noise_model, 'tq', 'N/A'),
                'measure_error': getattr(noise_model, 'measure', 'N/A'),
                'reset_error': getattr(noise_model, 'reset', 'N/A')
            }
        }

    if 'heron' in noise_types:
        results['custom_noise'] = results['heron_noise']

    results['comparison'] = _generate_comparison_summary_ibm(results)
    
    return results

def _process_full_experiment_results(
    job_id: str,
    ibm_results: Dict = None,
    circuits: List[QuantumCircuit] = None,
    circuit_metadata: List[Dict] = None
) -> Dict[str, Any]:

    if ibm_results is None:
        ibm_results = load_results_from_file(job_id)

        if ibm_results is None:
            print(f"Retrieving results for experiment job {job_id}...")
            ibm_results = get_ibm_job_results(job_id)

            results_file = f"job_results_{job_id}.json"
            with open(results_file, 'w') as f:
                json.dump(ibm_results, f, indent=2)
            print(f"Results saved to: {results_file}")
        
        if 'error' in ibm_results:
            print(f"Error retrieving results: {ibm_results['error']}")
            return {'error': ibm_results['error']}

    if circuits is None or circuit_metadata is None:
        job_info = load_job_info_from_file(job_id)
        if 'circuit_qasm' in job_info:
            try:
                circuits = [QuantumCircuit.from_qasm_str(qasm) for qasm in job_info['circuit_qasm']]
            except Exception as e:
                circuits = []
        if 'experiment_metadata' in job_info:
            circuit_metadata = job_info['experiment_metadata']['circuit_metadata']
        else:
            circuit_metadata = None

    if not circuits or not circuit_metadata:
        circuits, circuit_metadata = _recreate_experimental_circuits(
            qubit_range=(2, 7),
            circuits_per_qubit_size=10,
            circuit_depth=100,
            seed=42 
        )
    
    counts_list = ibm_results['counts']
    backend_name = ibm_results['metadata'].get('backend_name', 'unknown')

    results_by_qubit_size = {}
    overall_results = {
        'noiseless': {'fidelities': [], 'avg_fidelity': 0},
        'ibm_hardware': {'fidelities': [], 'avg_fidelity': 0},
        'heron_model': {'fidelities': [], 'avg_fidelity': 0},
        'modsi1000_001_model': {'fidelities': [], 'avg_fidelity': 0},
        'modsi1000_002_model': {'fidelities': [], 'avg_fidelity': 0},
        'job_metadata': ibm_results['metadata']
    }
    
    for i, ibm_counts in enumerate(counts_list):
        if i < len(circuits) and circuit_metadata and i < len(circuit_metadata):
            circuit = circuits[i]
            metadata = circuit_metadata[i]
            qubit_size = metadata['qubit_size']
            
            print(f"Analyzing circuit {i+1}/{len(counts_list)}: {qubit_size} qubits")

            if qubit_size not in results_by_qubit_size:
                results_by_qubit_size[qubit_size] = {
                    'noiseless': {'fidelities': []},
                    'ibm_hardware': {'fidelities': []},
                    'heron_model': {'fidelities': []},
                    'modsi1000_001_model': {'fidelities': []},
                    'modsi1000_002_model': {'fidelities': []},
                    'circuits': []
                }

            comparison_results = compare_circuit_execution_with_ibm_cloud(
                circuit=circuit,
                ibm_job_results=ibm_counts,
                circuit_index=i,
                noise_types=["heron", "modsi1000:0.001", "modsi1000:0.002"],
                noise_param=0.001,
                ibm_backend_name=backend_name
            )

            for backend_key in ['noiseless', 'ibm_hardware', 'heron_noise', 'modsi1000:0.001_noise', 'modsi1000:0.002_noise']:
                if backend_key in comparison_results:
                    if backend_key == 'noiseless':
                        fidelity = 1.0
                    else:
                        fidelity = _calculate_fidelity(
                            comparison_results['noiseless']['counts'],
                            comparison_results[backend_key]['counts']
                        )

                    if backend_key == 'heron_noise':
                        result_key = 'heron_model'
                    elif backend_key == 'modsi1000:0.001_noise':
                        result_key = 'modsi1000_001_model'
                    elif backend_key == 'modsi1000:0.002_noise':
                        result_key = 'modsi1000_002_model'
                    else:
                        result_key = backend_key
                    
                    results_by_qubit_size[qubit_size][result_key]['fidelities'].append(fidelity)
                    overall_results[result_key]['fidelities'].append(fidelity)
            
            results_by_qubit_size[qubit_size]['circuits'].append(circuit)
        
        else:
            
            ibm_fidelity = _calculate_ground_state_fidelity(ibm_counts)
            overall_results['ibm_hardware']['fidelities'].append(ibm_fidelity)
            
            print(f"  IBM Hardware ground state prob: {ibm_fidelity:.4f}")
    
    for qubit_size in sorted(results_by_qubit_size.keys()):
        results = results_by_qubit_size[qubit_size]
        
        for backend_key in ['noiseless', 'ibm_hardware', 'heron_model', 'modsi1000_001_model', 'modsi1000_002_model']:
            fidelities = results[backend_key]['fidelities']
            if fidelities:
                avg_fid = np.mean(fidelities)
                std_fid = np.std(fidelities)
                results[backend_key]['avg_fidelity'] = avg_fid
                results[backend_key]['std_fidelity'] = std_fid
                
    
    for backend_key in ['noiseless', 'ibm_hardware', 'heron_model', 'modsi1000_001_model', 'modsi1000_002_model']:
        fidelities = overall_results[backend_key]['fidelities']
        if fidelities:
            overall_results[backend_key]['avg_fidelity'] = np.mean(fidelities)
            overall_results[backend_key]['std_fidelity'] = np.std(fidelities)
    
    return {
        'results_by_qubit_size': results_by_qubit_size,
        'overall_results': overall_results,
        'job_metadata': ibm_results['metadata']
    }


def plot_experimental_results(results: Dict[str, Any], save_path: str = None) -> str:

    tex_fonts = {
        "font.family": "serif",
        "axes.labelsize": 12,
        "font.size": 12,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "axes.titlesize": 10,
        "lines.linewidth": 2,
        "lines.markersize": 6,
        "lines.markeredgewidth": 1.5,
        "lines.markeredgecolor": "black",
        "errorbar.capsize": 3,
    }
    plt.rcParams.update(tex_fonts)
    
    qubit_sizes = sorted([q for q in results['results_by_qubit_size'].keys() if q >= 3])
    
    palette = sns.color_palette("pastel", n_colors=4)
    models = {
        'ibm_hardware': {'label': 'IBM Hardware', 'color': palette[0], 'alpha': 1.0},
        'heron_model': {'label': 'Heron Model', 'color': palette[1], 'alpha': 1.0},
        'modsi1000_001_model': {'label': 'SI1000 (p=0.001)', 'color': palette[2], 'alpha': 1.0},
        'modsi1000_002_model': {'label': 'SI1000 (p=0.002)', 'color': palette[3], 'alpha': 1.0}
    }
    
    data = {model: {'means': [], 'stds': []} for model in models}
    
    for qubits in qubit_sizes:
        qubit_data = results['results_by_qubit_size'][qubits]
        
        for model_key in models:
            if model_key in qubit_data and qubit_data[model_key]['fidelities']:
                data[model_key]['means'].append(qubit_data[model_key].get('avg_fidelity', 0))
                data[model_key]['stds'].append(qubit_data[model_key].get('std_fidelity', 0))
            else:
                data[model_key]['means'].append(0)
                data[model_key]['stds'].append(0)
    
    
    fig, ax = plt.subplots(figsize=(5, 2.4))  
    
    x = np.array(qubit_sizes)
    n_models = len(models)
    bar_width = 0.15  
    
    bars = {}
    for i, (model_key, config) in enumerate(models.items()):
        offset = (i - (n_models-1)/2) * bar_width
        bars[model_key] = ax.bar(x + offset, data[model_key]['means'], bar_width, 
                                yerr=data[model_key]['stds'], 
                                label=config['label'], 
                                alpha=config['alpha'], 
                                capsize=3,
                                color=config['color'],
                                edgecolor='black',  
                                linewidth=1)
    
    ax.set_xlabel('Circuit Width', fontsize=12)
    ax.set_ylabel('Fidelity', fontsize=12)
    ax.set_title('Noise Comparison', 
                 fontsize=12, fontweight='bold', loc='left')  
    ax.set_xticks(x)
    ax.set_xticklabels(qubit_sizes)
    ax.legend(fontsize=10, loc='lower left', ncol=2)
    ax.grid(axis='y', alpha=0.3)  
    ax.set_axisbelow(True)  
    ax.set_ylim(0, 1.0)
    

    for spine in ax.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(1)
    
    plt.tight_layout()
    
    if save_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = f"experimental_results_multi_model_{timestamp}.pdf"
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight', format='pdf')
    plt.show()
    
    print(f"Plot saved to: {save_path}")
    return save_path


def main():
    job_id = "d5d5dj1smlfc739o1620"
    result = _process_full_experiment_results(job_id=job_id)
    plot_experimental_results(result)


if __name__ == "__main__":
    main()