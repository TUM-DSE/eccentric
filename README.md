# ECCentric Bench

Eccentric Bench is a project focused on realistic benchmarking Quantum Error Correction Codes. The preprint of our paper can be found here: https://arxiv.org/abs/2511.01062

## Installation (Linux)

Follow these steps to set up the project locally:

1. **Clone the repository:**

    ```bash
    git clone git@github.com:aswierkowska/eccentric_bench.git
    cd eccentric_bench
    ```

2. **Install `virtualenv`:**

    ```bash
    pip install virtualenv
    ```

3. **Create a virtual environment:**

    ```bash
    virtualenv venv
    ```

4. **Activate the virtual environment:**

    ```bash
    source venv/bin/activate
    ```

5. **Install the required dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

6. **Initialize the submodule:**

    ```bash
    git submodule update --init --recursive
    ```

7. **Build qiskit-qec**
    ```bash
    cd external/qiskit_qec
    python setup.py build_ext --inplace
    ```

## Execution

Once the environment is set up, you can run the project with:

```bash
python3 main.py experiments_conf/size.yaml
```


The results will be saved in the `experiments_result/`.

## Directory Structure

The repository is organized as follows:

- **`backends/`**: Configuration files and definitions for pseudo-realistic and artificial quantum backends.
- **`codes/`**: Scripts for generating Stim circuit files for Quantum Error Correction (QEC) codes.
- **`data/`**: Generated plots.
- **`decoders/`**: Implementations of decoding algorithms.
- **`experiments_conf/`**: YAML configuration files used to define and control experiments.
- **`experiments_results/`**: The default output directory where experiment results are saved.
- **`experiments_scripts/`**: Scripts for running specific types of experiments and generating statistics.
- **`metrics/`**: Scripts for calculating error correction metrics.
- **`noise/`**: Configuration files for pseudo-realistic and artificial noise models.
- **`plots/`**: Scripts for visualizing experiment results.
- **`transpilers/`**: Modules for circuit transpilation and translation between formats.