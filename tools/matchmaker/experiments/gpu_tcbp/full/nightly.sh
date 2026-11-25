#!/usr/bin/env bash
# tcbp_jobs_sc_gpu on the circuits of Matchmaker's table outside the GPU slice (list.json, build_list.py), error
# rates only, on steve's GPU at night and only while nobody else uses it:
#  - circuits start only between START and END o'clock Berlin time (default 22-07) and when no other user has a
#    process on the GPU; otherwise it waits
#  - a running circuit is stopped at END o'clock or as soon as another user's GPU process appears, and is redone
#    from the start later (results are written only when a circuit finishes)
#  - a circuit that runs past its jobs' time budget + 30 min is marked results/<i>.timeout and not retried; one that
#    crashes (no result) 3 times is marked results/<i>.failed
# Runs until every circuit has a result or a timeout mark. Detached: setsid nohup ./nightly.sh > logs/nightly.out &
# Stop: touch STOP (stops the running circuit and exits). Log: logs/nightly.log
HERE=/home/manosgior/eccentric-validation/gpu_dev/tcbp_full
SLICE_DIR=/home/manosgior/eccentric-validation/gpu_dev/tcbp_slice
PY=/home/manosgior/eccentric-validation/gpu_dev/py.sh
START=${START:-22}
END=${END:-7}
ME=$(id -un)
cd $HERE && mkdir -p results logs
log() { echo "$(date -u +%FT%TZ) $*" >> logs/nightly.log; }

in_window() {  # Berlin hour in [START, 24) or [0, END)
  local h=$((10#$(TZ=Europe/Berlin date +%H)))
  [ $h -ge $START ] || [ $h -lt $END ]
}
others_on_gpu() {  # prints the other users' GPU processes, if any
  nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | while read -r pid; do
    [ -n "$pid" ] || continue
    u=$(ps -o user= -p "$pid" 2>/dev/null | tr -d ' ')
    [ -n "$u" ] && [ "$u" != "$ME" ] && echo "$pid:$u"
  done
}
budget() {  # time budget (s) of circuit $1's CPU decode jobs
  python3 - "$1" <<'EOF'
import glob, json, os, sys
o = json.load(open("list.json"))[int(sys.argv[1])]
run, _, cid = o["key"].partition("/")
j = sorted(glob.glob(os.path.join("/home/manosgior/eccentric-validation", run, "jobs", "decode" + cid[len("circuit"):] + ".decoder=*.json")))[0]
print(json.load(open(j)).get("time_budget", 3600))
EOF
}

N=$(python3 -c "import json; print(len(json.load(open('list.json'))))")
log "start: $N circuits, window $START-$END Berlin"
while :; do
  [ -f STOP ] && { log "STOP file: exiting"; exit 0; }
  i=""
  for ((k=0; k<N; k++)); do
    [ -f results/$k.json ] || [ -f results/$k.timeout ] || [ -f results/$k.failed ] || { i=$k; break; }
  done
  [ -z "$i" ] && { log "all $N circuits done"; touch logs/nightly_done; exit 0; }
  if ! in_window; then sleep 300; continue; fi
  busy=$(others_on_gpu)
  if [ -n "$busy" ]; then log "GPU in use by $busy: waiting"; sleep 300; continue; fi
  limit=$(( $(budget $i) + 1800 ))
  log "start $i (limit ${limit}s)"
  t0=$(date +%s)
  CORES=192 TCBP_LIST=$HERE/list.json TCBP_RESULTS=$HERE/results TCBP_TIMING=0 \
    setsid $PY $SLICE_DIR/slice_circuit.py $i tcbp_jobs_sc_gpu >> logs/circuit_$((i / 100)).log 2>&1 &
  job=$!
  why=""
  while kill -0 $job 2>/dev/null; do
    sleep 30
    if [ -f STOP ]; then why="STOP file"
    elif ! in_window; then why="end of the night window"
    elif [ -n "$(others_on_gpu)" ]; then why="another user on the GPU: $(others_on_gpu | tr '\n' ' ')"
    elif [ $(( $(date +%s) - t0 )) -gt $limit ]; then why="timeout"
    fi
    if [ -n "$why" ]; then
      kill -TERM -- -$job 2>/dev/null; sleep 10; kill -KILL -- -$job 2>/dev/null
      break
    fi
  done
  wait $job 2>/dev/null
  if [ -f results/$i.json ]; then log "done $i in $(( $(date +%s) - t0 ))s"
  elif [ "$why" = "timeout" ]; then echo "$limit" > results/$i.timeout; log "timeout $i after ${limit}s: not retried"
  elif [ -n "$why" ]; then log "stopped $i ($why): redone later"
  else
    echo x >> results/$i.crashes
    if [ $(wc -l < results/$i.crashes) -ge 3 ]; then mv results/$i.crashes results/$i.failed; log "failed $i 3 times: not retried"
    else log "failed $i (no result; see logs/circuit_$((i / 100)).log): redone later"; sleep 60; fi
  fi
done
