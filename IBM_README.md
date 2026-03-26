# IBM Cloud Integration for Noise Model Comparison

This extension adds IBM Quantum cloud integration to the noise model comparison pipeline, allowing you to compare your custom noise models against real IBM hardware results.

## Overview

The enhanced pipeline now compares three backends:
1. **Noiseless simulator** - Perfect quantum computer (baseline)
2. **IBM Cloud hardware** - Real quantum hardware results
3. **Custom noise model** - Your proposed noise model

## Setup

### 1. IBM Quantum Account Setup
1. Create an account at https://quantum-computing.ibm.com/
2. Get your API token from account settings
3. Configure credentials:
   ```bash
   qiskit-ibm-runtime configure --channel ibm_quantum --token YOUR_TOKEN
   ```

### 2. Verify Setup
```python
python ibm_config.py
```

## Quick Start

### Method 1: Using the Demo Script
```bash
# List available backends
python ibm_demo.py backends

# Submit a test job
python ibm_demo.py submit

# Check job status
python ibm_demo.py status <job_id>

# Analyze results
python ibm_demo.py analyze <job_id>

# Run full pipeline (submit + wait + analyze)
python ibm_demo.py full
```

### Method 2: Direct Function Calls
```python
from noise_model_comparison import compare_random_circuits_with_ibm_cloud

# Submit circuits (returns immediately)
result = compare_random_circuits_with_ibm_cloud(
    num_qubits=3,
    depth=4,
    num_circuits=5,
    shots=1024,
    seed=42,
    submit_only=True
)
job_id = result['job_info']['job_id']

# Later: analyze results
analysis = compare_random_circuits_with_ibm_cloud(
    job_id=job_id,
    noise_type="heron"
)
```

## Key Features

### 🔧 **Batch Job Submission**
- Submit multiple circuits as a single job to minimize queue time
- Automatic shot optimization to respect IBM cloud limits
- Fixed seed for reproducible circuit generation

### 📊 **Separated Workflow**
- **Submit phase**: Generate circuits and submit to IBM cloud
- **Analysis phase**: Retrieve results and compare with custom models

### 🎯 **Smart Backend Selection**
- Automatic selection of best available backend
- Queue-aware recommendations
- Support for preferred backend lists

### 📈 **Comprehensive Analysis**
- Fidelity comparison between hardware and custom models
- Statistical analysis across multiple circuits
- Performance metrics and recommendations

## Usage Examples

### Basic Comparison
```python
# Generate 30 random 3-qubit circuits (10 per qubit) and submit to IBM
result = compare_random_circuits_with_ibm_cloud(
    num_qubits=3,
    depth=5,
    num_circuits=30,  # 10 circuits per qubit
    shots=1000,
    seed=42,
    backend_name="ibm_torino",
    submit_only=True
)
```

### Analysis of Results
```python
# Analyze completed job
analysis = compare_random_circuits_with_ibm_cloud(
    job_id="your_job_id_here",
    noise_type="heron",
    noise_param=0.001
)

# Access results
ibm_fidelity = analysis['ibm_hardware']['avg_fidelity']
model_fidelity = analysis['custom_noise']['avg_fidelity']
print(f"Model accuracy: {abs(ibm_fidelity - model_fidelity):.4f}")
```

### Backend Management
```python
from noise_model_comparison import get_available_ibm_backends, select_best_ibm_backend

# List all backends
backends = get_available_ibm_backends(min_qubits=3)
for b in backends:
    print(f"{b['name']}: {b['num_qubits']} qubits, queue: {b['pending_jobs']}")

# Auto-select best backend
best_backend = select_best_ibm_backend(min_qubits=3)
```

## Configuration

### Job Settings
Adjust settings in `ibm_config.py`:
```python
DEFAULT_JOB_SETTINGS = {
    'shots_per_circuit': 1000,       # Shots per circuit
    'optimization_level': 1,         # Transpilation optimization
    'max_circuits_per_job': 50,      # Circuits per job
    'max_shots_per_job': 100000,     # Total shots per job
    'default_timeout': 3600          # Job timeout (seconds)
}
```

### Backend Preferences
```python
DEFAULT_BACKEND_PREFERENCES = [
    "ibm_torino",
    "ibm_heron", 
    "ibm_flamingo",
    # Add your preferred backends
]
```

## Output and Results

### Job Submission Output
```
Job submitted successfully!
Job ID: 12345abcde
Backend: ibm_torino
Total shots: 5120
Estimated execution time: 5.1 min

Job information saved to: job_info_12345abcde.json
```

### Analysis Output
```
SUMMARY ACROSS 5 CIRCUITS
========================================
Noiseless: 1.0000 ± 0.0000
IBM ibm_torino: 0.7234 ± 0.0523
Custom (heron): 0.7456 ± 0.0487

Model Comparison:
  Custom model overestimates hardware performance
  Difference: 0.0222
  Relative error: 3.1%
```

## File Structure

```
├── noise_model_comparison.py    # Main comparison functions (enhanced)
├── ibm_config.py               # Configuration settings
├── ibm_demo.py                 # Demo and example usage
├── job_info_*.json             # Saved job information
└── experiment_results/         # Analysis results
```

## Cost Considerations

### Shot Optimization
- The system automatically optimizes shots to fit IBM limits
- Default: 1024 shots per circuit (good balance of accuracy vs. cost)
- For initial testing: Use 512 shots
- For production: Consider up to 2048 shots for better statistics

### Job Batching
- Multiple circuits are submitted as a single job
- Reduces queue waiting time
- More efficient use of allocated quantum time

### Example Cost Estimation
```python
from noise_model_comparison import estimate_job_cost

cost = estimate_job_cost(
    num_circuits=30,  # 10 circuits per qubit for 3-qubit example
    shots_per_circuit=1000,
    backend_name="ibm_torino"
)
print(cost)
```

## Error Handling

The system includes comprehensive error handling:
- Credential validation
- Backend availability checking  
- Job submission failures
- Result retrieval errors
- Network connectivity issues

## Best Practices

1. **Start Small**: Test with 3-5 circuits first
2. **Use Fixed Seeds**: Ensures reproducible results
3. **Monitor Queue**: Check backend status before submission
4. **Separate Phases**: Submit jobs, then analyze results later
5. **Save Job IDs**: Always save job information for later retrieval
6. **Validate Results**: Check job completion status before analysis

## Troubleshooting

### Common Issues

**"No IBM backends available"**
- Check your IBM Quantum account credentials
- Verify account has access to quantum systems
- Run `python ibm_config.py` to validate setup

**"Job failed or timeout"**
- Check job status with `python ibm_demo.py status <job_id>`
- Some jobs may take hours depending on queue
- Retry with fewer circuits or different backend

**"Cannot retrieve job results"**
- Ensure job has completed successfully
- Check if job ID is correct
- Verify network connectivity

### Getting Help

1. Run diagnostics: `python ibm_demo.py config`
2. Check IBM Quantum account status online
3. Verify network connectivity to IBM services
4. Check the saved job files: `job_info_*.json`

## Migration from FakeTorino

The original comparison still works with `compare_circuit_execution()`. The new IBM cloud functionality is accessed through `compare_random_circuits_with_ibm_cloud()`.

```python
# Old way (still works)
result = compare_circuit_execution(circuit, shots=1024)

# New way with IBM cloud
job_info = compare_random_circuits_with_ibm_cloud(..., submit_only=True)
result = compare_random_circuits_with_ibm_cloud(job_id=job_info['job_id'])
```