import os
from typing import Dict, List, Optional

DEFAULT_BACKEND_PREFERENCES = [
    "ibm_torino",
    "ibm_heron",
    "ibm_flamingo",
    "ibm_brisbane",
    "ibm_sherbrooke"
]

DEFAULT_JOB_SETTINGS = {
    'shots_per_circuit': 1000,
    'optimization_level': 1,
    'max_circuits_per_job': 60,
    'max_shots_per_job': 100000,
    'default_timeout': 7200
}

NOISE_MODEL_SETTINGS = {
    'default_noise_type': 'heron',
    'available_noise_types': ['heron', 'flamingo', 'infleqtion', 'artificial']
}

CIRCUIT_SETTINGS = {
    'default_seed': 42,
    'qubit_range': (2, 7),
    'circuits_per_qubit_size': 10,
    'default_depth': 100,
    'max_qubits': 7,
    'max_depth': 100
}

def get_config() -> Dict:
    return {
        'backends': DEFAULT_BACKEND_PREFERENCES,
        'job_settings': DEFAULT_JOB_SETTINGS,
        'noise_models': NOISE_MODEL_SETTINGS,
        'circuits': CIRCUIT_SETTINGS
    }

def validate_ibm_credentials() -> bool:
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
