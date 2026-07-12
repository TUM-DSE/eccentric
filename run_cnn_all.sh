#!/bin/bash
set -u
cd /qec
ts(){ date +%H:%M:%S; }
step(){ echo "===== $(ts) START $* ====="; }
LER_MODE=current    python -u run_ler_noise_models.py;    echo "===== $(ts) done current ====="
LER_MODE=futuristic python -u run_ler_noise_models.py;    echo "===== $(ts) done futuristic ====="
LER_MODE=highfid    python -u run_ler_noise_models.py;    echo "===== $(ts) done highfid ====="
python -u explore_readout.py;                             echo "===== $(ts) done explore ====="
MEMORY_ROUNDS=dplus1 python -u compare_readout_eccentric.py; echo "===== $(ts) done eccentric-dplus1 ====="
MEMORY_ROUNDS=1      python -u compare_readout_eccentric.py; echo "===== $(ts) done eccentric-nmr1 ====="
python -u evaluate_mcm_latency.py;                        echo "===== $(ts) done evaluate_mcm ====="
echo "===== $(ts) ALL ECCENTRIC-REPO RUNS COMPLETE ====="
