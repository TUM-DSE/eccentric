#!/usr/bin/env bash
# GPU runner: circuits in slice order, one at a time, each in its own process with a hard timeout
HERE=/home/manosgior/eccentric-validation/gpu_dev/mm_pilot
cd $HERE && mkdir -p results logs
N=$(python3 -c "import json; print(len(json.load(open('slice.json'))))")
for ((i=0; i<N; i++)); do
  [ -f results/$i.jsonl ] && grep -q '"kind": "done"' results/$i.jsonl && continue
  while [ ! -f prep/$i.pkl ]; do sleep 10; done
  CORES=192 timeout 3600 /home/manosgior/eccentric-validation/gpu_dev/py.sh $HERE/pilot_circuit.py $i >> logs/pilot_$i.log 2>&1
  echo "{\"kind\": \"done\", \"index\": $i, \"exit\": $?}" >> results/$i.jsonl
  rm -f prep/$i.pkl
done
echo finished > logs/runner_done
