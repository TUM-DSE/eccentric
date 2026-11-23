#!/usr/bin/env bash
# tcbp_jobs_sc_gpu over the GPU slice after runner.sh finishes (logs/runner_done), one circuit at a time (core 192)
HERE=/home/manosgior/eccentric-validation/gpu_dev/tcbp_slice
cd $HERE && mkdir -p results_tcbp_jobs_sc_gpu logs
while [ ! -f logs/runner_done ]; do sleep 60; done
N=$(python3 -c "import json; print(len(json.load(open('slice.json'))))")
for ((i=0; i<N; i++)); do
  [ -f results_tcbp_jobs_sc_gpu/$i.json ] && continue
  echo "$(date -u +%FT%TZ) start $i" >> logs/runner_sc.log
  CORES=192 timeout 7200 /home/manosgior/eccentric-validation/gpu_dev/py.sh $HERE/slice_circuit.py $i tcbp_jobs_sc_gpu >> logs/circuit_sc_$i.log 2>&1
  echo "$(date -u +%FT%TZ) end $i exit $?" >> logs/runner_sc.log
done
echo finished > logs/runner_sc_done
