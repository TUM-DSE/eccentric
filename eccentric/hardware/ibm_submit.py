import sys
import json
import os
import random
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from qiskit import QuantumCircuit, transpile
from qiskit_ibm_runtime import QiskitRuntimeService, Sampler


def get_available_ibm_backends(
    min_qubits: int = 2,
    operational_only: bool = True
) -> List[Dict[str, Any]]:
    try:
        service = QiskitRuntimeService()
        backends = service.backends()

        available = []
        for backend in backends:
            if backend.num_qubits >= min_qubits:
                status = backend.status()
                if not operational_only or status.operational:
                    backend_info = {
                        'name': backend.name,
                        'num_qubits': backend.num_qubits,
                        'operational': status.operational,
                        'pending_jobs': status.pending_jobs,
                        'status_msg': status.status_msg
                    }
                    try:
                        backend_info['queue_length'] = getattr(status, 'queue_length', 'N/A')
                    except Exception:
                        backend_info['queue_length'] = 'N/A'
                    available.append(backend_info)

        available.sort(key=lambda x: (
            x['pending_jobs'] if x['pending_jobs'] is not None else 999,
            -x['num_qubits']
        ))

        return available

    except Exception as e:
        print(f"Error fetching IBM backends: {e}")
        return []


def select_best_ibm_backend(
    min_qubits: int = 2,
    preferred_backends: List[str] = None
) -> Optional[str]:
    if preferred_backends is None:
        preferred_backends = ["ibm_torino", "ibm_heron", "ibm_flamingo"]

    available = get_available_ibm_backends(min_qubits=min_qubits)

    if not available:
        print("No available IBM backends found")
        return None

    print("\nAvailable IBM backends:")
    for backend in available:
        print(f"  {backend['name']}: {backend['num_qubits']} qubits, "
              f"queue: {backend['pending_jobs']}, operational: {backend['operational']}")

    for preferred in preferred_backends:
        for backend in available:
            if backend['name'] == preferred and backend['operational']:
                print(f"\nSelected preferred backend: {preferred}")
                return preferred

    for backend in available:
        if backend['operational']:
            print(f"\nSelected best available backend: {backend['name']}")
            return backend['name']

    print("No operational backends found")
    return None


def validate_ibm_credentials() -> bool:
    try:
        service = QiskitRuntimeService()
        backends = service.backends()
        return len(backends) > 0
    except Exception as e:
        print(f"IBM credentials not properly configured: {e}")
        print("Please run: qiskit-ibm-runtime configure --channel ibm_quantum --token YOUR_TOKEN")
        return False


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
            if random.random() < 0.7:
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


def submit_circuits_to_ibm_cloud(
    circuits: List[QuantumCircuit],
    backend_name: str = "ibm_torino",
    shots_per_circuit: int = 1024,
    optimization_level: int = 1,
    job_name: Optional[str] = None,
    save_job_info: bool = True,
    max_shots_per_job: int = 100000
) -> Dict[str, Any]:

    try:
        service = QiskitRuntimeService()
        backend = service.backend(backend_name)
    except Exception as e:
        print(f"Error connecting to IBM cloud: {e}")
        raise

    total_requested_shots = len(circuits) * shots_per_circuit
    if total_requested_shots > max_shots_per_job:
        adjusted_shots = max_shots_per_job // len(circuits)
        print(f"WARNING: Reducing shots from {shots_per_circuit} to {adjusted_shots} to fit IBM limits")
        shots_per_circuit = adjusted_shots

    measured_circuits = []
    for circuit in circuits:
        if not any(instr.operation.name == 'measure' for instr in circuit.data):
            circuit_with_measurements = circuit.copy()
            circuit_with_measurements.measure_all()
            circuit = circuit_with_measurements
        measured_circuits.append(circuit)

    transpiled_circuits = transpile(
        measured_circuits,
        backend=backend,
        optimization_level=optimization_level
    )

    if job_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        job_name = f"noise_comparison_{len(circuits)}circuits_{timestamp}"

    print(f"Submitting job '{job_name}' to {backend_name}...")

    try:
        sampler = Sampler(backend)
        job = sampler.run(transpiled_circuits, shots=shots_per_circuit)
    except Exception as e:
        raise RuntimeError(f"Could not submit job to {backend_name}. Error: {e}")

    job_info = {
        'job_id': job.job_id(),
        'backend_name': backend_name,
        'job_name': job_name,
        'num_circuits': len(circuits),
        'shots_per_circuit': shots_per_circuit,
        'total_shots': len(circuits) * shots_per_circuit,
        'submission_time': datetime.now().isoformat(),
        'optimization_level': optimization_level,
        'circuit_depths': [c.depth() for c in transpiled_circuits],
        'circuit_sizes': [c.size() for c in transpiled_circuits],
    }

    print(f"Job ID: {job.job_id()}")

    if save_job_info:
        job_file = f"experiment_results/ibm_runs/job_info_{job.job_id()}.json"
        serializable_job_info = job_info.copy()
        try:
            serializable_job_info['circuit_qasm'] = [c.qasm() for c in transpiled_circuits]
        except AttributeError:
            serializable_job_info['circuit_qasm'] = [str(c) for c in transpiled_circuits]

        with open(job_file, 'w') as f:
            json.dump(serializable_job_info, f, indent=2)
        print(f"Job information saved to: {job_file}")

    return job_info


def get_ibm_job_results(
    job_id: str,
    wait_for_completion: bool = True,
    timeout: int = 3600
) -> Dict[str, Any]:

    service = QiskitRuntimeService()

    try:
        job = service.job(job_id)
        print(f"Retrieved job {job_id}")

        job_status = job.status()
        if hasattr(job_status, 'name'):
            status_name = job_status.name
        else:
            status_name = str(job_status)

        print(f"Job status: {status_name}")

        if wait_for_completion:
            print(f"Waiting for job completion (timeout: {timeout}s)...")
            result = job.result(timeout=timeout)
        else:
            if status_name not in ['DONE', 'CANCELLED', 'ERROR']:
                print("Job not completed yet. Set wait_for_completion=True to wait.")
                return {'status': status_name, 'job_id': job_id}
            result = job.result()

        counts_list = []
        if hasattr(result, '__len__'):
            for i in range(len(result)):
                counts_list.append(result[i].data.meas.get_counts())
        else:
            counts_list.append(result.data.meas.get_counts())

        metadata = {
            'job_id': job_id,
            'status': status_name,
            'backend_name': job.backend().name,
            'creation_time': job.creation_date.isoformat() if job.creation_date else None,
            'completion_time': datetime.now().isoformat(),
            'num_circuits': len(counts_list),
        }

        try:
            metadata.update({
                'queue_position': getattr(job, 'queue_position', None),
                'execution_time': getattr(job, 'time_taken', None),
            })
        except Exception:
            pass

        return {
            'metadata': metadata,
            'counts': counts_list,
        }

    except Exception as e:
        print(f"Error retrieving job results: {e}")
        return {'error': str(e), 'job_id': job_id}


def load_job_info_from_file(job_id: str) -> Dict[str, Any]:
    job_file = f"experiment_results/ibm_runs/job_info_{job_id}.json"
    if os.path.exists(job_file):
        with open(job_file, 'r') as f:
            return json.load(f)
    else:
        print(f"Job info file not found: {job_file}")
        return {}


def load_results_from_file(job_id: str) -> Optional[Dict[str, Any]]:
    results_file = f"experiment_results/ibm_runs/job_results_{job_id}.json"
    if os.path.exists(results_file):
        print(f"Loading results from local file: {results_file}")
        with open(results_file, 'r') as f:
            return json.load(f)
    return None


def run_full_experiment_submit(
    qubit_range: Tuple[int, int] = (2, 7),
    circuits_per_qubit_size: int = 10,
    circuit_depth: int = 100,
    shots: int = 1000,
    seed: int = 42,
    backend_name: str = None,
) -> Dict[str, Any]:

    qubit_sizes = list(range(qubit_range[0], qubit_range[1] + 1))
    total_circuits = len(qubit_sizes) * circuits_per_qubit_size

    random.seed(seed)
    np.random.seed(seed)

    all_circuits = []
    circuit_metadata = []

    for qubit_size in qubit_sizes:
        for i in range(circuits_per_qubit_size):
            circuit = generate_random_circuit_with_torino_gates(
                num_qubits=qubit_size,
                depth=circuit_depth,
                seed=seed + qubit_size * 1000 + i
            )
            all_circuits.append(circuit)
            circuit_metadata.append({
                'qubit_size': qubit_size,
                'circuit_index': i,
                'depth': circuit.depth(),
                'size': circuit.size()
            })

    if backend_name is None:
        backend_name = select_best_ibm_backend(min_qubits=max(qubit_sizes))
        if backend_name is None:
            raise RuntimeError(f"No suitable IBM backend available for {max(qubit_sizes)} qubits")

    job_info = submit_circuits_to_ibm_cloud(
        circuits=all_circuits,
        backend_name=backend_name,
        shots_per_circuit=shots,
        job_name=f"noise_experiment_{total_circuits}circuits_depth{circuit_depth}_seed{seed}"
    )

    job_info['experiment_metadata'] = {
        'qubit_range': qubit_range,
        'circuits_per_qubit_size': circuits_per_qubit_size,
        'circuit_depth': circuit_depth,
        'circuit_metadata': circuit_metadata,
        'qubit_sizes': qubit_sizes
    }

    job_file = f"experiment_results/ibm_runs/job_info_{job_info['job_id']}.json"
    serializable = {k: v for k, v in job_info.items() if k != 'transpiled_circuits'}
    with open(job_file, 'w') as f:
        json.dump(serializable, f, indent=2, default=str)

    return job_info


def retrieve_and_save_results(job_id: str) -> Dict[str, Any]:

    existing = load_results_from_file(job_id)
    if existing is not None:
        print(f"Results already saved locally for job {job_id}")
        return existing

    print(f"Retrieving results for job {job_id}...")
    ibm_results = get_ibm_job_results(job_id, wait_for_completion=True)

    if 'error' in ibm_results:
        print(f"Error retrieving results: {ibm_results['error']}")
        return ibm_results

    results_file = f"experiment_results/ibm_runs/job_results_{job_id}.json"
    with open(results_file, 'w') as f:
        json.dump(ibm_results, f, indent=2)
    print(f"Results saved to: {results_file}")

    return ibm_results


def submit_job():

    if not validate_ibm_credentials():
        print("Please configure IBM credentials")
        return None

    backends = get_available_ibm_backends(min_qubits=7)
    if not backends:
        print("No suitable backends available!")
        return None

    print(f"\nCOST WARNING:")
    print(f"  This will submit 60 circuits x 1000 shots = 60,000 total shots")

    response = input("\nProceed with submission? (yes/no): ").lower()
    if response not in ['yes', 'y']:
        print("Submission cancelled.")
        return None

    print(f"\nSubmitting experiment...")

    job_info = run_full_experiment_submit(
        qubit_range=(2, 7),
        circuits_per_qubit_size=10,
        circuit_depth=100,
        shots=1000,
        seed=42,
    )

    job_id = job_info['job_id']
    print(f"Job ID: {job_id}")

    print(f"To retrieve results: python -m eccentric.hardware.ibm_submit results {job_id}")

    return job_id


def get_results(job_id: str):
    print("Trying to get results for job ID:", job_id)

    result = retrieve_and_save_results(job_id)

    if 'error' in result:
        print(f"Error: {result['error']}")
        return None

    print(f"Number of circuits: {result['metadata']['num_circuits']}")
    print(f"Backend: {result['metadata']['backend_name']}")

    return result


def main():

    if len(sys.argv) < 2:
        print("Usage: python -m eccentric.hardware.ibm_submit submit | results <job_id>")
        return

    command = sys.argv[1].lower()

    if command == "submit":
        submit_job()

    elif command == "results" and len(sys.argv) > 2:
        job_id = sys.argv[2]
        get_results(job_id)

    else:
        print("Unknown command. Available commands:")
        print("  submit                  - Submit experiment to IBM cloud")
        print("  results <job_id>        - Retrieve and save job results")


if __name__ == "__main__":
    main()
