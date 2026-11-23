#!/usr/bin/env bash
# tcbp_jobs_gpu over the GPU slice, one circuit at a time, each in its own process with a hard timeout (core 192)
HERE=/home/manosgior/eccentric-validation/gpu_dev/tcbp_slice
cd $HERE && mkdir -p results logs
N=$(python3 -c "import json; print(len(json.load(open('slice.json'))))")
for ((i=0; i<N; i++)); do
  [ -f results/$i.json ] && continue
  echo "$(date -u +%FT%TZ) start $i" >> logs/runner.log
  CORES=192 timeout 7200 /home/manosgior/eccentric-validation/gpu_dev/py.sh $HERE/slice_circuit.py $i >> logs/circuit_$i.log 2>&1
  echo "$(date -u +%FT%TZ) end $i exit $?" >> logs/runner.log
done
echo finished > logs/runner_done
