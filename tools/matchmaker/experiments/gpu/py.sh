#!/usr/bin/env bash
# GPU Python on steve: nix python, gpu_venv (+ main venv packages), NVIDIA driver libs, CPU pinned to cores 192-255
cd /home/manosgior/eccentric-validation/gpu_repo
exec nix-shell .nix_env/shell.nix --run "LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:/run/opengl-driver/lib OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONWARNINGS=ignore PYTHONPATH=/home/manosgior/eccentric-validation/gpu_repo:/home/manosgior/eccentric-validation/gpu_dev/verify_code taskset -c ${CORES:-192-255} ../gpu_venv/bin/python $*" </dev/null
