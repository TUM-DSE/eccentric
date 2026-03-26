FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy the requirements file first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
# Using --no-cache-dir to keep the image size down
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project
# Note: This assumes that you have already run `git submodule update --init --recursive`
# on your host machine before building the docker image.
COPY . .

# Build Qiskit-QEC
RUN cd external/qiskit_qec && python setup.py build_ext --inplace

# By default, running the docker container will sequentially run all experiments and plots
CMD ["bash", "run_experiments.sh"]
