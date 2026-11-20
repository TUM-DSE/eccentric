#!/usr/bin/env bash
# 4 CPU prep workers (cores 200-203) and the GPU runner (core 192), detached
HERE=/home/manosgior/eccentric-validation/gpu_dev/mm_pilot
cd $HERE && mkdir -p results logs prep claims
for c in 200 201 202 203; do
  (CORES=$c setsid nohup /home/manosgior/eccentric-validation/gpu_dev/py.sh $HERE/prep.py > logs/prep_$c.log 2>&1 < /dev/null &)
done
(setsid nohup $HERE/runner.sh > logs/runner.log 2>&1 < /dev/null &)
