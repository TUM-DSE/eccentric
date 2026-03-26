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

**Build Qiskit-QEC:**
```bash
cd external/qiskit_qec
python setup.py build_ext --inplace
```


## Running the Project

Once the environment is set up, you can run the project with:

```bash
python3 main.py <yaml_file>
```

The Docker setup automatically runs a full suite of experiments (size, connectivity, variance, technology) and generates the corresponding plots. Results and logs will be saved throughout the project directory.

## Running with Docker

To ensure full reproducibility and avoid local environment issues, you can run the benchmark inside a Docker container.

1. **Build the Docker Image:**
     Make sure you have pulled the git submodules first (`git submodule update --init --recursive`), then build the image:
     ```bash
     docker build -t eccentric_bench .
     ``

2. **Run All Experiments and Generate Plots:**
     Running the container will automatically execute the full suite of experiments sequentially via `run_experiments.sh`. This process can take a significant amount of time depending on the parameters:
     ```bash
     docker run --name eccentric_bench_run eccentric_bench
     ```

3. **Extract Results:**
     Since all logs, experiment data, and plots are generated inside the container, the easiest way to extract them is to copy the entire application directory to your local machine:
     ```bash
     docker cp eccentric_bench_run:/app/eccentric_bench_results .
     ```
     Once completed, you can remove the stopped container with: `docker rm eccentric_bench_run`

## Possible Erros
**IBM API token:**
    This project may require using an IBM API token, which should be saved in your working environment. Please follow the instructions from the IBM guide to set up the token:

    [IBM Quantum API Setup Guide](https://docs.quantum.ibm.com/guides/setup-channel)