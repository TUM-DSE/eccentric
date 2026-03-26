# IBM Cloud Configuration for Noise Model Comparison
# 
# This file contains settings for IBM Quantum cloud execution.
# Make sure you have properly set up your IBM Quantum account and saved your credentials.
# 
# To set up IBM Quantum credentials:
# 1. Create an account at https://quantum-computing.ibm.com/
# 2. Get your API token from your account settings
# 3. Run: qiskit-ibm-runtime configure --channel ibm_quantum --token YOUR_TOKEN
# 
# Or create a ~/.qiskit/qiskit-ibm.json file with your credentials.

import os
from typing import Dict, List, Optional

# Default IBM backend preferences (in order of preference)
DEFAULT_BACKEND_PREFERENCES = [
    "ibm_torino",
    "ibm_heron", 
    "ibm_flamingo",
    "ibm_brisbane",
    "ibm_sherbrooke"
]

# Default job settings
DEFAULT_JOB_SETTINGS = {
    'shots_per_circuit': 1000,
    'optimization_level': 1,
    'max_circuits_per_job': 60,  # To handle full experimental batch
    'max_shots_per_job': 100000,  # IBM cloud limit (60 * 1000 = 60,000 fits)
    'default_timeout': 7200  # 2 hour timeout for larger jobs
}

# Noise model settings for comparison
NOISE_MODEL_SETTINGS = {
    'default_noise_type': 'heron',  # Heron noise model (no parameters needed)
    'available_noise_types': ['heron', 'flamingo', 'infleqtion', 'artificial']
}

# Circuit generation settings
CIRCUIT_SETTINGS = {
    'default_seed': 42,
    'qubit_range': (2, 7),  # 2 to 7 qubits (6 different sizes)
    'circuits_per_qubit_size': 10,  # 10 circuits per qubit size = 60 total
    'default_depth': 100,  # Depth 100 for all circuits
    'max_qubits': 7,
    'max_depth': 100
}

def get_config() -> Dict:
    """Get configuration settings."""
    return {
        'backends': DEFAULT_BACKEND_PREFERENCES,
        'job_settings': DEFAULT_JOB_SETTINGS,
        'noise_models': NOISE_MODEL_SETTINGS,
        'circuits': CIRCUIT_SETTINGS
    }

def validate_ibm_credentials() -> bool:
    """Check if IBM Quantum credentials are properly configured."""
    try:
        from qiskit_ibm_runtime import QiskitRuntimeService
        service = QiskitRuntimeService()
        backends = service.backends()
        return len(backends) > 0
    except Exception as e:
        print(f"IBM credentials not properly configured: {e}")
        print("Please run: qiskit-ibm-runtime configure --channel ibm_quantum --token YOUR_TOKEN")
        return False

if __name__ == "__main__":
    config = get_config()
    print("IBM Cloud Configuration:")
    for section, settings in config.items():
        print(f"\n{section.upper()}:")
        if isinstance(settings, dict):
            for key, value in settings.items():
                print(f"  {key}: {value}")
        elif isinstance(settings, list):
            for i, item in enumerate(settings):
                print(f"  [{i}]: {item}")
        else:
            print(f"  {settings}")
    
    print(f"\nCredentials valid: {validate_ibm_credentials()}")