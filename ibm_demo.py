#!/usr/bin/env python3
"""
IBM Cloud Noise Model Comparison Demo

This script demonstrates how to use the new IBM cloud functionality 
for comparing noise models with real hardware.

Usage examples:
1. Submit circuits to IBM cloud:
   python ibm_demo.py submit

2. Check job status:
   python ibm_demo.py status <job_id>

3. Analyze completed job:
   python ibm_demo.py analyze <job_id>

4. Run full pipeline (submit and wait):
   python ibm_demo.py full

5. List available backends:
   python ibm_demo.py backends
"""

import sys
import time
from noise_model_comparison import (
    run_full_noise_experiment_ibm_cloud,
    get_available_ibm_backends,
    select_best_ibm_backend,
    estimate_job_cost,
    get_ibm_job_results,
    plot_experimental_results
)
from ibm_config import validate_ibm_credentials, get_config


def demo_submit_job():
    """Demo: Submit the full 60-circuit experiment to IBM cloud."""
    print("=" * 70)
    print("SUBMITTING FULL NOISE EXPERIMENT TO IBM CLOUD")
    print("=" * 70)
    print("This will submit 60 circuits:")
    print("  • 10 circuits each for 2, 3, 4, 5, 6, 7 qubits")
    print("  • Depth 100, 1000 shots per circuit")
    print("  • Total: 60,000 shots")
    print("  • Compare IBM hardware vs. Heron noise model")
    
    # Check credentials
    if not validate_ibm_credentials():
        print("Please configure IBM credentials first!")
        return None
    
    # Show available backends
    backends = get_available_ibm_backends(min_qubits=7)  # Need 7 qubits for experiment
    if not backends:
        print("No suitable backends available (need at least 7 qubits)!")
        return None
    
    print(f"\nAvailable backends (7+ qubits):")
    for b in backends[:3]:  # Show top 3
        print(f"  {b['name']}: {b['num_qubits']} qubits, queue: {b['pending_jobs']}")
    
    # Cost warning
    print(f"\n⚠️  COST WARNING:")
    print(f"  This will submit 60 circuits × 1000 shots = 60,000 total shots")
    print(f"  This uses significant IBM Quantum credits!")
    
    response = input("\nProceed with submission? (yes/no): ").lower()
    if response not in ['yes', 'y']:
        print("Submission cancelled.")
        return None
    
    # Submit the full experiment
    print(f"\nSubmitting full noise experiment...")
    
    result = run_full_noise_experiment_ibm_cloud(
        qubit_range=(2, 7),  # 2 to 7 qubits
        circuits_per_qubit_size=10,  # 10 per size
        circuit_depth=100,  # Depth 100
        shots=1000,
        seed=42,
        submit_only=True
    )
    
    job_id = result['job_info']['job_id']
    print(f"\n✓ Full experiment submitted successfully!")
    print(f"Job ID: {job_id}")
    print(f"Backend: {result['job_info']['backend_name']}")
    print(f"Total circuits: {result['job_info']['num_circuits']}")
    print(f"Total shots: {result['job_info']['total_shots']:,}")
    
    print(f"\nTo check status: python ibm_demo.py status {job_id}")
    print(f"To analyze results: python ibm_demo.py analyze {job_id}")
    
    return job_id


def demo_check_status(job_id: str):
    """Demo: Check the status of an IBM job."""
    print("=" * 60)
    print(f"DEMO: CHECKING JOB STATUS")
    print("=" * 60)
    
    try:
        from qiskit_ibm_runtime import QiskitRuntimeService
        service = QiskitRuntimeService()
        job = service.job(job_id)
        
        # Handle different ways job status might be returned
        job_status = job.status()
        if hasattr(job_status, 'name'):
            status_name = job_status.name
        else:
            status_name = str(job_status)
        
        print(f"Job ID: {job_id}")
        print(f"Status: {status_name}")
        print(f"Backend: {job.backend().name}")
        
        if hasattr(job, 'queue_position') and job.queue_position():
            print(f"Queue position: {job.queue_position()}")
        
        if status_name == 'DONE':
            print("Job is complete! Ready for analysis.")
        elif status_name in ['QUEUED', 'RUNNING']:
            print("Job is still running. Check back later.")
        else:
            print(f"Job status: {status_name}")
            
    except Exception as e:
        print(f"Error checking job status: {e}")


def demo_analyze_results(job_id: str):
    """Demo: Analyze results from the full 60-circuit experiment."""
    print("=" * 70)
    print("ANALYZING FULL EXPERIMENT RESULTS")
    print("=" * 70)
    
    print(f"Retrieving and analyzing results for job: {job_id}")
    
    result = run_full_noise_experiment_ibm_cloud(job_id=job_id)
    
    if 'error' in result:
        print(f"Error: {result['error']}")
        return None
    
    print("\n✓ Analysis completed!")
    
    # Show summary
    if 'overall_results' in result:
        overall = result['overall_results']
        
        if ('ibm_hardware' in overall and 'heron_model' in overall and
            overall['ibm_hardware']['fidelities'] and overall['heron_model']['fidelities']):
            
            ibm_avg = overall['ibm_hardware']['avg_fidelity']
            heron_avg = overall['heron_model']['avg_fidelity']
            
            print(f"\nEXPERIMENT SUMMARY (60 circuits, depth 100):")
            print(f"  IBM Hardware: {ibm_avg:.4f} ± {overall['ibm_hardware']['std_fidelity']:.4f}")
            print(f"  Heron Model:  {heron_avg:.4f} ± {overall['heron_model']['std_fidelity']:.4f}")
            
            diff = abs(heron_avg - ibm_avg)
            print(f"  Model accuracy: {diff:.4f} difference ({diff/ibm_avg*100:.1f}% relative error)")
            
            if heron_avg > ibm_avg:
                print("  → Heron model OVERESTIMATES hardware performance")
            elif heron_avg < ibm_avg:
                print("  → Heron model UNDERESTIMATES hardware performance") 
            else:
                print("  → Heron model matches hardware performance!")
            
            # Show breakdown by qubit size
            if 'results_by_qubit_size' in result:
                print(f"\n  Results by qubit size:")
                for qubits in sorted(result['results_by_qubit_size'].keys()):
                    qdata = result['results_by_qubit_size'][qubits]
                    if 'ibm_hardware' in qdata and qdata['ibm_hardware']['fidelities']:
                        hw_avg = qdata['ibm_hardware']['avg_fidelity']
                        model_avg = qdata['heron_model']['avg_fidelity']
                        print(f"    {qubits} qubits: HW={hw_avg:.3f}, Model={model_avg:.3f}")
    
    # Generate plot
    if 'results_by_qubit_size' in result:
        print(f"\n📊 Generating plot...")
        plot_path = plot_experimental_results(result)
        if plot_path:
            print(f"✓ Plot saved: {plot_path}")
    
    return result


def demo_full_pipeline():
    """Demo: Run the complete experimental pipeline (submit and analyze)."""
    print("=" * 70)
    print("FULL EXPERIMENTAL PIPELINE")
    print("=" * 70)
    print("This will:")
    print("  1. Submit 60 circuits (10 each for 2-7 qubits, depth 100)")
    print("  2. Wait for completion (may take hours)")
    print("  3. Analyze IBM hardware vs. Heron noise model")
    
    # Submit job
    job_id = demo_submit_job()
    if job_id is None:
        return
    
    print(f"\nWaiting for job completion...")
    print("This may take several hours depending on queue...")
    
    # Wait and analyze
    result = run_full_noise_experiment_ibm_cloud(job_id=job_id)
    
    if 'error' not in result:
        print("\n✓ Full experimental pipeline completed successfully!")
    else:
        print(f"Pipeline failed: {result['error']}")


def demo_list_backends():
    """Demo: List available IBM backends."""
    print("=" * 60)
    print("DEMO: AVAILABLE IBM BACKENDS")
    print("=" * 60)
    
    if not validate_ibm_credentials():
        return
    
    backends = get_available_ibm_backends()
    
    print(f"Found {len(backends)} backends:\n")
    print(f"{'Name':<20} {'Qubits':<8} {'Queue':<8} {'Status'}")
    print("-" * 50)
    
    for backend in backends:
        status_icon = "✓" if backend['operational'] else "✗"
        print(f"{backend['name']:<20} {backend['num_qubits']:<8} {backend['pending_jobs']:<8} {status_icon}")
    
    # Show recommendation
    best = select_best_ibm_backend(min_qubits=2)
    if best:
        print(f"\nRecommended backend: {best}")


def main():
    """Main demo function."""
    
    if len(sys.argv) < 2:
        print(__doc__)
        return
    
    command = sys.argv[1].lower()
    
    if command == "submit":
        demo_submit_job()
        
    elif command == "status" and len(sys.argv) > 2:
        job_id = sys.argv[2]
        demo_check_status(job_id)
        
    elif command == "analyze" and len(sys.argv) > 2:
        job_id = sys.argv[2]
        demo_analyze_results(job_id)
        
    elif command == "full":
        demo_full_pipeline()
        
    elif command == "backends":
        demo_list_backends()
        
    elif command == "config":
        config = get_config()
        print("Current configuration:")
        for section, settings in config.items():
            print(f"\n{section.upper()}:")
            for key, value in settings.items():
                print(f"  {key}: {value}")
        print(f"\nCredentials valid: {validate_ibm_credentials()}")
        
    else:
        print("Unknown command. Available commands:")
        print("  submit, status <job_id>, analyze <job_id>, full, backends, config")


if __name__ == "__main__":
    main()