from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_ibm_runtime import Estimator, Sampler
from qiskit import QuantumCircuit, transpile
from qiskit_ibm_runtime.fake_provider import FakeTorino
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel as QiskitNoiseModel
from qiskit.result import Result
from qiskit.circuit.library import XGate, SXGate, RZGate, CXGate, IGate
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import random
import matplotlib.pyplot as plt
import json
import os
from datetime import datetime

from noise import get_noise_model
from noise.heron_noise import HeronNoise
from backends import QubitTracking


def compare_circuit_execution(
    circuit: QuantumCircuit,
    shots: int = 1024,
    noise_type: str = "heron", 
    noise_param: float = 0.001,
    custom_backend=None,
    custom_noise_model=None
) -> Dict[str, Any]:
    """
    Compare circuit execution across three different backends:
    1. Noiseless simulator
    2. FakeTorino backend (IBM's fake backend with built-in noise)
    3. Custom backend with provided noise model
    
    IMPORTANT: All backends execute the EXACT same transpiled circuit for fair comparison.
    
    Parameters:
    -----------
    circuit : QuantumCircuit
        The quantum circuit to execute
    shots : int
        Number of shots for execution (default: 1024)
    noise_type : str
        Type of noise model to use ('heron', 'flamingo', 'infleqtion', etc.)
    noise_param : float  
        Noise parameter (if applicable for the noise type)
    custom_backend : BackendV2, optional
        Custom backend to use (if None, will use a default based on noise_type)
    custom_noise_model : NoiseModel, optional
        Custom noise model to override the default
        
    Returns:
    --------
    Dict containing results from all three backends with execution statistics
    """
    
    results = {}
    
    # Ensure circuit has measurements
    if not any(instr.operation.name == 'measure' for instr in circuit.data):
        # Add measurements to all qubits
        circuit_with_measurements = circuit.copy()
        circuit_with_measurements.add_register(circuit_with_measurements.cregs[0] if circuit_with_measurements.cregs else circuit_with_measurements.add_register('c', circuit.num_qubits))
        circuit_with_measurements.measure_all()
        circuit = circuit_with_measurements
    
    # CRITICAL: Transpile once to get identical circuit for all backends
    fake_torino = FakeTorino()
    
    # For Heron devices, we should use CZ instead of CX
    # Transpile to a common basis set that both backends can handle
    common_transpiled = transpile(
        circuit, 
        fake_torino,
        basis_gates=['sx', 'x', 'rz', 'cz', 'id'],  # Use CZ for Heron compatibility
        optimization_level=1
    )
    
    print(f"\nTranspiled circuit depth: {common_transpiled.depth()}")
    print(f"Transpiled circuit gates: {common_transpiled.count_ops()}")
    
    # 1. Noiseless Simulator
    print("Running on noiseless simulator...")
    noiseless_simulator = AerSimulator()
    # Use the same transpiled circuit for fair comparison
    noiseless_job = noiseless_simulator.run(common_transpiled, shots=shots)
    noiseless_result = noiseless_job.result()
    
    results['noiseless'] = {
        'backend': 'AerSimulator (noiseless)',
        'counts': noiseless_result.get_counts(),
        'fidelity': 1.0,  # Noiseless is always perfect fidelity
        'execution_time': getattr(noiseless_result, 'time_taken', 'N/A')
    }
    
    # 2. FakeTorino Backend
    print("Running on FakeTorino backend...")
    
    # Create simulator with FakeTorino's noise model
    torino_noise_model = QiskitNoiseModel.from_backend(fake_torino)
    
    # Extract noise parameters for comparison
    torino_noise_details = _extract_torino_noise_parameters(torino_noise_model, fake_torino)
    
    torino_simulator = AerSimulator(noise_model=torino_noise_model)
    
    # Use the same transpiled circuit for fair comparison
    torino_job = torino_simulator.run(common_transpiled, shots=shots)
    torino_result = torino_job.result()
    
    # Calculate fidelity relative to noiseless
    torino_fidelity = _calculate_fidelity(
        results['noiseless']['counts'], 
        torino_result.get_counts()
    )
    
    results['fake_torino'] = {
        'backend': 'FakeTorino with noise',
        'counts': torino_result.get_counts(),
        'fidelity': torino_fidelity,
        'execution_time': getattr(torino_result, 'time_taken', 'N/A'),
        'coupling_map': fake_torino.coupling_map,
        'num_qubits': fake_torino.num_qubits,
        'torino_noise_details': torino_noise_details
    }
    
    # 3. Custom Backend with Custom Noise Model
    print(f"Running on custom backend with {noise_type} noise...")
    
    # Set up backend first
    if custom_backend is None:
        custom_backend = fake_torino  # Use Torino architecture as base
    
    # Set up QubitTracking with proper parameters using the common transpiled circuit
    qt = QubitTracking(custom_backend, common_transpiled)
    
    # Get the appropriate noise model
    if noise_type == "heron":
        noise_model = HeronNoise.get_noise(qt, custom_backend)
    else:
        # Use the generic noise model getter
        noise_model = get_noise_model(noise_type, qt, noise_param, custom_backend)
    
    if custom_noise_model is not None:
        noise_model = custom_noise_model
    
    # Create a Qiskit noise model approximation that uses CZ instead of CX for Heron
    qiskit_custom_noise = _create_qiskit_noise_approximation(noise_model, custom_backend, use_cz=True)
    custom_simulator = AerSimulator(noise_model=qiskit_custom_noise)
    
    # Use the same transpiled circuit for fair comparison
    custom_job = custom_simulator.run(common_transpiled, shots=shots)
    custom_result = custom_job.result()
    
    # Calculate fidelity relative to noiseless
    custom_fidelity = _calculate_fidelity(
        results['noiseless']['counts'],
        custom_result.get_counts()
    )
    
    results['custom_noise'] = {
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
    
    # Summary comparison
    results['comparison'] = _generate_comparison_summary(results)
    
    return results


def _calculate_success_probability(counts: Dict[str, int]) -> float:
    """Calculate success probability. For Bell states, success is '00' or '11'."""
    total_shots = sum(counts.values())
    if total_shots == 0:
        return 0.0
    
    # Get the bit string length
    if not counts:
        return 0.0
    
    sample_key = list(counts.keys())[0]
    
    # For Bell states (2 qubits), success is '00' or '11'
    if ' ' in sample_key:  # Format like '00 00' or '11 00'
        # For Bell state, we want correlated outcomes: both qubits same value
        success_states = ['00 00', '11 00']  # Perfect Bell state outcomes
        success_count = sum(counts.get(state, 0) for state in success_states)
    elif len(sample_key) == 2:  # Simple 2-qubit case without spaces
        # Bell state success: both qubits same value
        success_states = ['00', '11']
        success_count = sum(counts.get(state, 0) for state in success_states)
    else:
        # For other circuits, assume ground state is target
        ground_state = '0' * len(sample_key)
        success_count = counts.get(ground_state, 0)
    
    return success_count / total_shots


def _calculate_fidelity(ideal_counts: Dict[str, int], noisy_counts: Dict[str, int]) -> float:
    """Calculate fidelity between two count distributions using Total Variation distance.
    
    This is the proper metric for comparing random circuit results.
    Fidelity = 1 - 0.5 * sum(|p_ideal(x) - p_noisy(x)|)
    
    Args:
        ideal_counts: Counts from noiseless simulation
        noisy_counts: Counts from noisy simulation
        
    Returns:
        Fidelity between 0 and 1 (1 = identical distributions)
    """
    # Get total shots for normalization
    total_ideal = sum(ideal_counts.values())
    total_noisy = sum(noisy_counts.values())
    
    if total_ideal == 0 or total_noisy == 0:
        return 0.0
    
    # Get all possible outcomes
    all_outcomes = set(ideal_counts.keys()) | set(noisy_counts.keys())
    
    # Calculate Total Variation distance
    tv_distance = 0.0
    for outcome in all_outcomes:
        p_ideal = ideal_counts.get(outcome, 0) / total_ideal
        p_noisy = noisy_counts.get(outcome, 0) / total_noisy
        tv_distance += abs(p_ideal - p_noisy)
    
    # Convert TV distance to fidelity
    fidelity = 1.0 - 0.5 * tv_distance
    return max(0.0, fidelity)  # Ensure non-negative


def _extract_torino_noise_parameters(torino_noise_model, fake_torino_backend):
    """Extract noise parameters from FakeTorino's noise model for comparison."""
    noise_details = {
        'gate_errors': {},
        'readout_errors': {},
        'reset_errors': {},
        't1_times': {},
        't2_times': {}
    }
    
    try:
        # Try to get backend properties from FakeTorino directly
        if hasattr(fake_torino_backend, 'properties'):
            props = fake_torino_backend.properties()
            
            # Single-qubit gate errors (average across qubits)
            sx_errors = []
            x_errors = []
            rz_errors = []
            
            for qubit in range(fake_torino_backend.num_qubits):
                try:
                    # Get single-qubit gate errors
                    if hasattr(props, 'gate_error'):
                        sx_error = props.gate_error('sx', qubit)
                        x_error = props.gate_error('x', qubit) 
                        rz_error = props.gate_error('rz', qubit)
                        
                        sx_errors.append(sx_error)
                        x_errors.append(x_error)
                        rz_errors.append(rz_error)
                except:
                    continue
            
            if sx_errors:
                noise_details['gate_errors']['sx_avg'] = np.mean(sx_errors)
                noise_details['gate_errors']['x_avg'] = np.mean(x_errors)
                noise_details['gate_errors']['rz_avg'] = np.mean(rz_errors)
            
            # Two-qubit gate errors (CZ and CX)
            cz_errors = []
            cx_errors = []
            coupling_map = fake_torino_backend.coupling_map
            if coupling_map:
                for edge in coupling_map.get_edges():
                    try:
                        # Try CZ first (Heron devices)
                        if hasattr(props, 'gate_error'):
                            try:
                                cz_error = props.gate_error('cz', edge)
                                cz_errors.append(cz_error)
                            except Exception as e:
                                pass
                            
                            try:
                                cx_error = props.gate_error('cx', edge)
                                cx_errors.append(cx_error)
                            except Exception as e:
                                pass
                    except:
                        continue
            
            # Filter out unrealistic errors (> 50% are likely disconnected qubits)
            if cz_errors:
                filtered_cz = [err for err in cz_errors if err < 0.5]  # Remove 100% error qubits
                if filtered_cz:
                    noise_details['gate_errors']['cz_avg'] = np.mean(filtered_cz)
                    
            if cx_errors:
                filtered_cx = [err for err in cx_errors if err < 0.5]  # Remove 100% error qubits
                if filtered_cx:
                    noise_details['gate_errors']['cx_avg'] = np.mean(filtered_cx)
            
            # Readout errors
            readout_errors = []
            for qubit in range(fake_torino_backend.num_qubits):
                try:
                    if hasattr(props, 'readout_error'):
                        ro_error = props.readout_error(qubit)
                        readout_errors.append(ro_error)
                except:
                    continue
            
            if readout_errors:
                noise_details['readout_errors']['average'] = np.mean(readout_errors)
            
            # Try to extract reset errors from the noise model
            if hasattr(torino_noise_model, '_default_quantum_errors'):
                errors = torino_noise_model._default_quantum_errors
                if 'reset' in errors:
                    reset_info = errors['reset']
                    if hasattr(reset_info, 'probabilities') and reset_info.probabilities:
                        # Reset error is typically the probability of incorrect reset
                        reset_prob = reset_info.probabilities[0] if reset_info.probabilities[0] else 0
                        noise_details['reset_errors']['average'] = reset_prob
                        
            # T1 and T2 times
            t1_times = []
            t2_times = []
            for qubit in range(fake_torino_backend.num_qubits):
                try:
                    t1 = props.t1(qubit)
                    t2 = props.t2(qubit)
                    if t1 is not None:
                        t1_times.append(t1)
                    if t2 is not None:
                        t2_times.append(t2)
                except:
                    continue
            
            if t1_times:
                noise_details['t1_times']['average'] = np.mean(t1_times)
            if t2_times:
                noise_details['t2_times']['average'] = np.mean(t2_times)
        
    except Exception as e:
        noise_details['extraction_error'] = f"Could not extract parameters: {e}"
        
        # Fallback: use typical IBM values for FakeTorino based on the device spec
        noise_details['gate_errors']['sx_avg'] = 0.0007    # From our extraction above
        noise_details['gate_errors']['x_avg'] = 0.0007     # From our extraction above
        noise_details['gate_errors']['rz_avg'] = 0.0       # Virtual Z gate
        noise_details['gate_errors']['cx_avg'] = 0.0108    # Typical IBM CNOT error for Torino-class
        noise_details['gate_errors']['cz_avg'] = 0.0108    # Typical IBM CZ error for Torino-class
        noise_details['readout_errors']['average'] = 0.047  # From our extraction above
        noise_details['reset_errors']['average'] = 0.002   # Typical IBM reset error
        noise_details['t1_times']['average'] = 174e-6      # From our extraction above  
        noise_details['t2_times']['average'] = 145e-6      # From our extraction above
    
    return noise_details


def print_noise_comparison(torino_details: Dict, heron_params: Dict):
    """Print side-by-side comparison of noise models."""
    print(f"{'Parameter':<25} {'FakeTorino':<20} {'Heron Model':<20} {'Ratio (H/T)':<15}")
    print("-" * 80)
    
    # Compare single-qubit errors
    torino_sq = torino_details.get('gate_errors', {}).get('sx_avg', 'N/A')
    heron_sq = heron_params.get('sq_error', 'N/A')
    ratio_sq = _calculate_ratio(heron_sq, torino_sq)
    print(f"{'Single-qubit error':<25} {_format_value(torino_sq):<20} {_format_value(heron_sq):<20} {ratio_sq:<15}")
    
    # Compare two-qubit errors (prefer CZ for Heron compatibility)
    torino_tq = torino_details.get('gate_errors', {}).get('cz_avg', 
                    torino_details.get('gate_errors', {}).get('cx_avg', 'N/A'))
    heron_tq = heron_params.get('tq_error', 'N/A')
    ratio_tq = _calculate_ratio(heron_tq, torino_tq)
    gate_name = 'CZ' if 'cz_avg' in torino_details.get('gate_errors', {}) else 'CX'
    print(f"{f'Two-qubit error ({gate_name})':<25} {_format_value(torino_tq):<20} {_format_value(heron_tq):<20} {ratio_tq:<15}")
    
    # Compare readout errors
    torino_ro = torino_details.get('readout_errors', {}).get('average', 'N/A')
    heron_ro = heron_params.get('measure_error', 'N/A') 
    ratio_ro = _calculate_ratio(heron_ro, torino_ro)
    print(f"{'Measurement error':<25} {_format_value(torino_ro):<20} {_format_value(heron_ro):<20} {ratio_ro:<15}")
    
    # Compare reset errors
    torino_reset = torino_details.get('reset_errors', {}).get('average', 'N/A')
    heron_reset = heron_params.get('reset_error', 'N/A')
    ratio_reset = _calculate_ratio(heron_reset, torino_reset)
    print(f"{'Reset error':<25} {_format_value(torino_reset):<20} {_format_value(heron_reset):<20} {ratio_reset:<15}")
    
    print()  # Empty line before coherence times
    # Show T1/T2 times if available
    if 't1_times' in torino_details and 'average' in torino_details['t1_times']:
        t1_val = torino_details['t1_times']['average']
        print(f"{'T1 time (μs)':<25} {t1_val*1e6:.1f}μs")
    
    if 't2_times' in torino_details and 'average' in torino_details['t2_times']:
        t2_val = torino_details['t2_times']['average']
        print(f"{'T2 time (μs)':<25} {t2_val*1e6:.1f}μs")


def _format_value(val):
    """Format numerical values for display."""
    if val == 'N/A' or val is None:
        return 'N/A'
    try:
        if isinstance(val, (int, float)):
            if val < 0.001:
                return f"{val:.2e}"
            else:
                return f"{val:.6f}"
        return str(val)
    except:
        return str(val)


def _calculate_ratio(val1, val2):
    """Calculate ratio between two values."""
    try:
        if val1 == 'N/A' or val2 == 'N/A' or val1 is None or val2 is None:
            return 'N/A'
        v1 = float(val1)
        v2 = float(val2)
        if v2 == 0:
            return 'N/A'
        ratio = v1 / v2
        return f"{ratio:.1f}x"
    except (ValueError, TypeError):
        return 'N/A'


def _create_qiskit_noise_approximation(custom_noise_model, backend, use_cz=False):
    """
    Create a simplified Qiskit noise model approximation of the custom noise model.
    This is a basic approximation since your noise models are Stim-based.
    """
    from qiskit_aer.noise import NoiseModel, depolarizing_error, ReadoutError
    
    noise_model = NoiseModel()
    
    # Add single-qubit depolarizing error
    if hasattr(custom_noise_model, 'sq') and custom_noise_model.sq > 0:
        sq_error = depolarizing_error(custom_noise_model.sq, 1)
        noise_model.add_all_qubit_quantum_error(sq_error, ['sx', 'x', 'rz'])
    
    # Add two-qubit depolarizing error (CZ for Heron, CX for others)
    if hasattr(custom_noise_model, 'tq') and custom_noise_model.tq > 0:
        tq_error = depolarizing_error(custom_noise_model.tq, 2)
        if use_cz:
            noise_model.add_all_qubit_quantum_error(tq_error, ['cz'])
        else:
            noise_model.add_all_qubit_quantum_error(tq_error, ['cx', 'cz'])
    
    # Add measurement error
    if hasattr(custom_noise_model, 'measure') and custom_noise_model.measure > 0:
        measure_error = ReadoutError([[1-custom_noise_model.measure, custom_noise_model.measure],
                                    [custom_noise_model.measure, 1-custom_noise_model.measure]])
        noise_model.add_all_qubit_readout_error(measure_error)
    
    # Add reset error if available
    if hasattr(custom_noise_model, 'reset') and custom_noise_model.reset > 0:
        reset_error = depolarizing_error(custom_noise_model.reset, 1)
        noise_model.add_all_qubit_quantum_error(reset_error, ['reset'])
    
    return noise_model


def _generate_comparison_summary(results: Dict[str, Any]) -> Dict[str, Any]:
    """Generate a summary comparing the different execution results."""
    summary = {
        'success_probabilities': {},
        'error_rates': {},
        'fidelity_comparison': {}
    }
    
    noiseless_fidelity = results['noiseless']['fidelity']
    
    for backend_key in ['noiseless', 'fake_torino', 'custom_noise']:
        if backend_key in results:
            fidelity = results[backend_key]['fidelity']
            summary['success_probabilities'][backend_key] = fidelity
            summary['error_rates'][backend_key] = 1 - fidelity
            
            # Calculate fidelity relative to noiseless case
            if noiseless_fidelity > 0:
                summary['fidelity_comparison'][backend_key] = fidelity / noiseless_fidelity
            else:
                summary['fidelity_comparison'][backend_key] = 0.0
    
    return summary


def print_comparison_results(results: Dict[str, Any]):
    """Pretty print the comparison results."""
    print("\n" + "="*60)
    print("QUANTUM CIRCUIT EXECUTION COMPARISON")
    print("="*60)
    
    for backend_key in ['noiseless', 'fake_torino', 'custom_noise']:
        if backend_key in results:
            result = results[backend_key]
            print(f"\n{result['backend'].upper()}:")
            print(f"  Fidelity: {result['fidelity']:.4f}")
            print(f"  Error Rate: {1-result['fidelity']:.4f}")
            print(f"  Top 3 outcomes: {dict(sorted(result['counts'].items(), key=lambda x: x[1], reverse=True)[:3])}")
            
            if 'noise_parameters' in result:
                print(f"  Noise Parameters:")
                for param, value in result['noise_parameters'].items():
                    print(f"    {param}: {value}")
            
            if 'torino_noise_details' in result:
                print(f"  FakeTorino Noise Details:")
                for param, value in result['torino_noise_details'].items():
                    print(f"    {param}: {value}")
    
    if 'comparison' in results:
        print(f"\nFIDELITY COMPARISON (relative to noiseless):")
        for backend, fidelity in results['comparison']['fidelity_comparison'].items():
            print(f"  {backend}: {fidelity:.4f}")
    
    # Add noise parameter comparison if both models available
    if 'fake_torino' in results and 'custom_noise' in results:
        if 'torino_noise_details' in results['fake_torino'] and 'noise_parameters' in results['custom_noise']:
            print(f"\n{'='*60}")
            print("NOISE MODEL COMPARISON")
            print("="*60)
            print_noise_comparison(
                results['fake_torino']['torino_noise_details'], 
                results['custom_noise']['noise_parameters']
            )


# Example usage function
def example_bell_state_comparison():
    """Example usage with a Bell state circuit."""
    # Create a Bell state circuit
    circuit = QuantumCircuit(2, 2)
    circuit.h(0)
    circuit.cx(0, 1)
    circuit.measure_all()
    
    # Run comparison
    results = compare_circuit_execution(
        circuit=circuit,
        shots=10000,
        noise_type="heron",
        noise_param=0.001
    )
    
    # Print results
    print_comparison_results(results)
    
    return results


def generate_random_circuit_with_torino_gates(
    num_qubits: int,
    depth: int,
    seed: Optional[int] = None
) -> QuantumCircuit:
    """
    Generate a random circuit using only FakeTorino's native gate set.
    
    Parameters:
    -----------
    num_qubits : int
        Number of qubits in the circuit
    depth : int  
        Circuit depth (number of gate layers)
    seed : int, optional
        Random seed for reproducibility
        
    Returns:
    --------
    QuantumCircuit with only FakeTorino-compatible gates
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    # FakeTorino's native gate set (check target for exact gates)
    fake_torino = FakeTorino()
    
    # Common IBM gate set - use CZ for Heron compatibility
    single_qubit_gates = ['x', 'sx', 'rz', 'id']  # X, SX, RZ, Identity
    two_qubit_gates = ['cz']  # Use CZ instead of CX for Heron compatibility
    
    circuit = QuantumCircuit(num_qubits, num_qubits)
    
    for layer in range(depth):
        # Randomly choose qubits to apply gates to
        available_qubits = list(range(num_qubits))
        random.shuffle(available_qubits)
        
        used_qubits = set()
        
        # Apply some two-qubit gates first
        num_two_qubit = random.randint(0, min(2, num_qubits // 2))
        for _ in range(num_two_qubit):
            if len(available_qubits) < 2:
                break
                
            # Choose two qubits that aren't already used
            available_pairs = [(i, j) for i in available_qubits 
                             for j in available_qubits 
                             if i != j and i not in used_qubits and j not in used_qubits]
            
            if not available_pairs:
                break
                
            qubit1, qubit2 = random.choice(available_pairs)
            
            # Apply CZ gate (Heron-compatible)
            circuit.cz(qubit1, qubit2)
            used_qubits.update([qubit1, qubit2])
        
        # Apply single-qubit gates to remaining qubits
        remaining_qubits = [q for q in available_qubits if q not in used_qubits]
        for qubit in remaining_qubits:
            if random.random() < 0.7:  # 70% chance to apply a gate
                gate_type = random.choice(single_qubit_gates)
                
                if gate_type == 'x':
                    circuit.x(qubit)
                elif gate_type == 'sx':
                    circuit.sx(qubit)
                elif gate_type == 'rz':
                    # Random rotation angle
                    angle = random.uniform(0, 2 * np.pi)
                    circuit.rz(angle, qubit)
                elif gate_type == 'id':
                    circuit.id(qubit)
        
        # Add a barrier between layers for clarity
        if layer < depth - 1:
            circuit.barrier()
    
    # Add measurements
    circuit.measure_all()
    
    return circuit


def compare_random_circuits(
    num_qubits: int = 3,
    depth: int = 5,
    num_circuits: int = 5,
    shots: int = 1024,
    noise_type: str = "heron",
    noise_param: float = 0.001,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Generate multiple random circuits and compare their execution across backends.
    
    Parameters:
    -----------
    num_qubits : int
        Number of qubits per circuit
    depth : int
        Depth of each random circuit
    num_circuits : int
        Number of random circuits to test
    shots : int
        Shots per circuit execution
    noise_type : str
        Type of custom noise model
    noise_param : float
        Noise parameter
    seed : int
        Random seed for reproducibility
        
    Returns:
    --------
    Dictionary with aggregated results across all circuits
    """
    
    print(f"\n{'='*60}")
    print(f"RANDOM CIRCUIT COMPARISON")
    print(f"Testing {num_circuits} random circuits ({num_qubits} qubits, depth {depth})")
    print(f"{'='*60}")
    
    all_results = {
        'noiseless': {'success_probs': [], 'avg_fidelity': 0},
        'fake_torino': {'success_probs': [], 'avg_fidelity': 0}, 
        'custom_noise': {'success_probs': [], 'avg_fidelity': 0},
        'circuits': []
    }
    
    for i in range(num_circuits):
        print(f"\n--- Circuit {i+1}/{num_circuits} ---")
        
        # Generate random circuit
        circuit = generate_random_circuit_with_torino_gates(
            num_qubits=num_qubits,
            depth=depth, 
            seed=seed + i  # Different seed for each circuit
        )
        
        all_results['circuits'].append(circuit)
        
        # Get ground state (all zeros) probability as success metric
        # For random circuits, we'll use ground state fidelity
        results = compare_circuit_execution(
            circuit=circuit,
            shots=shots,
            noise_type=noise_type,
            noise_param=noise_param
        )
        
        # Extract success probabilities  
        for backend_key in ['noiseless', 'fake_torino', 'custom_noise']:
            if backend_key in results:
                success_prob = _calculate_ground_state_fidelity(results[backend_key]['counts'])
                all_results[backend_key]['success_probs'].append(success_prob)
        
        # Print brief summary for this circuit
        print(f"  Noiseless: {all_results['noiseless']['success_probs'][-1]:.4f}")
        print(f"  FakeTorino: {all_results['fake_torino']['success_probs'][-1]:.4f}")  
        print(f"  Custom: {all_results['custom_noise']['success_probs'][-1]:.4f}")
    
    # Calculate averages
    for backend_key in ['noiseless', 'fake_torino', 'custom_noise']:
        if all_results[backend_key]['success_probs']:
            all_results[backend_key]['avg_fidelity'] = np.mean(all_results[backend_key]['success_probs'])
            all_results[backend_key]['std_fidelity'] = np.std(all_results[backend_key]['success_probs'])
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"SUMMARY ACROSS {num_circuits} RANDOM CIRCUITS")
    print(f"{'='*60}")
    
    for backend_key in ['noiseless', 'fake_torino', 'custom_noise']:
        backend_name = {
            'noiseless': 'Noiseless',
            'fake_torino': 'FakeTorino', 
            'custom_noise': f'Custom ({noise_type})'
        }[backend_key]
        
        if backend_key in all_results and all_results[backend_key]['success_probs']:
            avg = all_results[backend_key]['avg_fidelity']
            std = all_results[backend_key]['std_fidelity']
            print(f"{backend_name}: {avg:.4f} ± {std:.4f}")
    
    return all_results


def _calculate_ground_state_fidelity(counts: Dict[str, int]) -> float:
    """Calculate fidelity to ground state (all zeros)."""
    total_shots = sum(counts.values())
    if total_shots == 0:
        return 0.0
    
    # Look for ground state pattern
    sample_key = list(counts.keys())[0] if counts else ""
    
    if ' ' in sample_key:  # Format like '000 000'  
        ground_pattern = '0' * (len(sample_key.split()[0])) + ' ' + '0' * (len(sample_key.split()[1]))
    else:
        ground_pattern = '0' * len(sample_key)
    
    ground_count = counts.get(ground_pattern, 0)
    return ground_count / total_shots


# Example usage for random circuits
def example_random_circuit_comparison():
    """Example usage with random circuits."""
    results = compare_random_circuits(
        num_qubits=3,
        depth=4,
        num_circuits=3,
        shots=1024,
        noise_type="heron",
        seed=42
    )
    
    return results


def run_multi_circuit_comparison(
    num_qubits_range: Tuple[int, int] = (2, 7),
    circuits_per_qubit: int = 5,  # Reduced from 50 to 5
    shots: int = 1024,
    max_depth: int = 10,
    save_results: bool = True,
    results_file: str = None
) -> Dict[str, Any]:
    """Run comprehensive fidelity comparison across multiple qubit counts.
    
    Args:
        num_qubits_range: (min_qubits, max_qubits) range to test
        circuits_per_qubit: Number of random circuits per qubit count
        shots: Number of shots per circuit execution
        max_depth: Maximum circuit depth for random generation
        save_results: Whether to save results to file
        results_file: Custom filename for results (auto-generated if None)
    
    Returns:
        Dictionary containing fidelity statistics for each backend and qubit count
    """
    print("\n=== MULTI-CIRCUIT FIDELITY COMPARISON ===\n")
    
    fake_torino = FakeTorino()
    all_results = {
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'num_qubits_range': num_qubits_range,
            'circuits_per_qubit': circuits_per_qubit,
            'shots': shots,
            'max_depth': max_depth
        },
        'raw_data': {},  # Circuit-by-circuit results
        'statistics': {}  # Mean/std for each qubit count
    }
    
    for num_qubits in range(num_qubits_range[0], num_qubits_range[1] + 1):
        print(f"\n--- Testing {num_qubits}-qubit circuits ({circuits_per_qubit} circuits) ---")
        
        qubit_results = {
            'noiseless': [],
            'fake_torino': [],
            'heron': []
        }
        
        for i in range(circuits_per_qubit):
            # Progress tracking
            progress = i + 1
            if progress % max(1, circuits_per_qubit // 5) == 0 or progress == circuits_per_qubit:
                print(f"  Progress: {progress}/{circuits_per_qubit} circuits ({100*progress/circuits_per_qubit:.0f}%)")
            
            try:
                # Generate random circuit
                circuit_depth = min(max_depth, max(3, num_qubits * 2))
                circuit = generate_random_circuit_with_torino_gates(
                    num_qubits=num_qubits,
                    depth=circuit_depth,
                    seed=42 + num_qubits * 1000 + i
                )
                
                # Transpile for QubitTracking
                transpiled = transpile(circuit, backend=fake_torino, optimization_level=1)
                qt = QubitTracking(fake_torino, transpiled)
                heron_noise = HeronNoise.get_noise(qt, fake_torino)
                
                # Run comparison
                result = compare_circuit_execution(
                    circuit=circuit,
                    shots=shots,
                    custom_backend=fake_torino,
                    custom_noise_model=heron_noise
                )
                
                # Store fidelities (now using proper fidelity metric)
                qubit_results['noiseless'].append(result['noiseless']['fidelity'])
                qubit_results['fake_torino'].append(result['fake_torino']['fidelity']) 
                qubit_results['heron'].append(result['custom_noise']['fidelity'])
                
            except Exception as e:
                print(f"    Warning: Circuit {i+1} failed: {e}")
                continue
        
        # Calculate statistics
        stats = {}
        for backend_name, fidelities in qubit_results.items():
            if fidelities:
                stats[backend_name] = {
                    'mean': float(np.mean(fidelities)),
                    'std': float(np.std(fidelities)),
                    'min': float(np.min(fidelities)),
                    'max': float(np.max(fidelities)),
                    'count': len(fidelities)
                }
            else:
                stats[backend_name] = {'mean': 0, 'std': 0, 'min': 0, 'max': 0, 'count': 0}
        
        all_results['raw_data'][str(num_qubits)] = qubit_results
        all_results['statistics'][str(num_qubits)] = stats
        
        # Print summary for this qubit count
        print(f"  Results for {num_qubits} qubits:")
        for backend, stat in stats.items():
            if stat['count'] > 0:
                print(f"    {backend}: {stat['mean']:.3f} ± {stat['std']:.3f} (n={stat['count']})")
    
    # Save results if requested
    if save_results:
        if results_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_file = f"experiment_results/fidelity_comparison_{timestamp}.json"
        
        os.makedirs(os.path.dirname(results_file), exist_ok=True)
        with open(results_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"\nResults saved to: {results_file}")
    
    return all_results


def plot_fidelity_comparison(
    results: Dict[str, Any],
    save_plot: bool = True,
    plot_file: str = None,
    show_plot: bool = False
) -> None:
    """Create bar plot comparing fidelities across different qubit counts.
    
    Args:
        results: Results dictionary from run_multi_circuit_comparison
        save_plot: Whether to save plot to file
        plot_file: Custom filename for plot (auto-generated if None)
        show_plot: Whether to display plot interactively
    """
    # Extract data for plotting
    qubit_counts = sorted([int(k) for k in results['statistics'].keys()])
    backend_names = ['noiseless', 'fake_torino', 'heron']
    backend_labels = ['Noiseless', 'FakeTorino', 'Heron Model']
    colors = ['#2E8B57', '#DC143C', '#4169E1']  # Green, Red, Blue
    
    means = {backend: [] for backend in backend_names}
    stds = {backend: [] for backend in backend_names}
    
    for qubits in qubit_counts:
        stats = results['statistics'][str(qubits)]
        for backend in backend_names:
            means[backend].append(stats[backend]['mean'])
            stds[backend].append(stats[backend]['std'])
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Set bar width and positions
    bar_width = 0.25
    x_pos = np.arange(len(qubit_counts))
    
    # Create bars for each backend
    for i, (backend, label, color) in enumerate(zip(backend_names, backend_labels, colors)):
        bars = ax.bar(
            x_pos + i * bar_width,
            means[backend],
            bar_width,
            yerr=stds[backend],
            label=label,
            color=color,
            alpha=0.8,
            capsize=5
        )
    
    # Customize the plot
    ax.set_xlabel('Number of Qubits', fontsize=12, fontweight='bold')
    ax.set_ylabel('Average Fidelity', fontsize=12, fontweight='bold')
    ax.set_title('Quantum Circuit Fidelity Comparison\n(Error bars show standard deviation)', 
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos + bar_width)
    ax.set_xticklabels(qubit_counts)
    ax.legend(loc='upper right', fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, 1.05)
    
    # Add some statistics as text
    metadata = results['metadata']
    info_text = f"Circuits per qubit: {metadata['circuits_per_qubit']}\nShots per circuit: {metadata['shots']}"
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes, fontsize=9, 
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    # Save plot if requested
    if save_plot:
        if plot_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plot_file = f"plots/fidelity_comparison_{timestamp}.png"
        
        os.makedirs(os.path.dirname(plot_file), exist_ok=True)
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {plot_file}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()


def load_comparison_results(results_file: str) -> Dict[str, Any]:
    """Load previously saved comparison results from file.
    
    Args:
        results_file: Path to JSON results file
        
    Returns:
        Results dictionary
    """
    with open(results_file, 'r') as f:
        return json.load(f)


def run_complete_fidelity_study(
    num_qubits_range: Tuple[int, int] = (2, 7),
    circuits_per_qubit: int = 50,
    shots: int = 1024
) -> str:
    """Run the complete fidelity study and generate plots.
    
    Args:
        num_qubits_range: Range of qubit counts to test
        circuits_per_qubit: Number of circuits per qubit count
        shots: Shots per circuit
        
    Returns:
        Path to the generated plot file
    """
    print("\n" + "="*60)
    print("COMPREHENSIVE QUANTUM FIDELITY COMPARISON STUDY")
    print("="*60)
    
    # Run the multi-circuit comparison
    results = run_multi_circuit_comparison(
        num_qubits_range=num_qubits_range,
        circuits_per_qubit=circuits_per_qubit,
        shots=shots
    )
    
    # Generate the plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_file = f"plots/fidelity_study_{timestamp}.png"
    
    plot_fidelity_comparison(results, save_plot=True, plot_file=plot_file)
    
    # Print summary
    print("\n" + "="*60)
    print("STUDY SUMMARY")
    print("="*60)
    
    for qubits in sorted([int(k) for k in results['statistics'].keys()]):
        stats = results['statistics'][str(qubits)]
        print(f"\n{qubits}-qubit circuits:")
        for backend, stat in stats.items():
            if stat['count'] > 0:
                print(f"  {backend.replace('_', ' ').title()}: {stat['mean']:.3f} ± {stat['std']:.3f}")
    
    print(f"\nPlot saved to: {plot_file}")
    return plot_file


def quick_fidelity_test():
    """Quick test with fewer circuits for development/testing."""
    return run_complete_fidelity_study(
        num_qubits_range=(2, 4),
        circuits_per_qubit=5,
        shots=512
    )


def medium_fidelity_test():
    """Medium test with reasonable number of circuits."""
    return run_complete_fidelity_study(
        num_qubits_range=(2, 6),
        circuits_per_qubit=20,
        shots=1024
    )


# Quick usage examples:
# - Quick test: quick_fidelity_test()
# - Medium test: medium_fidelity_test()  
# - Full study: run_complete_fidelity_study()
# - Custom: run_complete_fidelity_study(num_qubits_range=(2,5), circuits_per_qubit=30)


if __name__ == "__main__":
    # Run the Bell state example
    print("=== BELL STATE COMPARISON ===")
    example_bell_state_comparison()
    
    print("\n\n=== RANDOM CIRCUIT COMPARISON ===")
    # Run the random circuit example
    example_random_circuit_comparison()




