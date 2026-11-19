# Comparison with real IBM hardware

`eccentric.hardware.ibm_submit` submits random circuits (2–7 qubits, depth 100, 10 circuits per width, 1000 shots each) to an IBM Quantum backend. `eccentric.hardware.ibm_demo` then compares the measured fidelities with noiseless simulation and the noise models in this repository.

```bash
python -m eccentric.hardware.ibm_submit submit            # submit the job (asks for confirmation; uses IBM credits)
python -m eccentric.hardware.ibm_submit results <job_id>  # download and store the results
python -m eccentric.hardware.ibm_demo [<job_id>]          # analyse and plot (defaults to the job used in the paper)
```

Job metadata and results are stored in `experiment_results/ibm_runs/`. These scripts need an IBM Quantum API token. Follow the [IBM Quantum setup guide](https://quantum.cloud.ibm.com/docs/en/guides/cloud-setup) to save it in your environment.
