#!/bin/bash
set -e

python3 -m eccentric \
    configs/size.yaml \
    configs/connectivity.yaml \
    configs/variance.yaml \
    configs/technologies.yaml \
    configs/dqc.yaml \
    configs/routing.yaml \
    configs/translation.yaml \
    configs/accumulated.yaml \
    configs/decoder.yaml \
    configs/chromobius.yaml \
    configs/program_stats.yaml

python3 -m eccentric.plots
python3 -m eccentric.hardware.ibm_demo
