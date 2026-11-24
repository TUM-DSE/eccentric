#!/usr/bin/env bash
# 24 Nov 2026, detached: after the tcbp_jobs_sc_gpu slice pass (runner_sc.sh) finishes,
#  1. refit the 21 TCBP models fitted on another run's circuit of the same name (tcbp_models.circuit_file fix)
#  2. rerun tcbp_jobs_sc_gpu on the 68 slice circuits affected (39 whose slice.json entry named the wrong circuit, and
#     the other circuits of the 21 codes); their earlier results (both decoders) move to stale_wrong_circuit/
#  3. redo the GPU pilots (../gpu, mm_pilot) of the 39 circuits on the corrected slice.json
# Log: logs/tonight.log
T=/home/manosgior/eccentric-validation/gpu_dev/tcbp_slice
P=/home/manosgior/eccentric-validation/gpu_dev/mm_pilot
PY=/home/manosgior/eccentric-validation/gpu_dev/py.sh
CODES="bb108 bb162 bb180 bb288 bb54 bpc126 bpc144 bpc162 bpc180 gb126 gb46 gb48 hgp225 lcs15 lcs150 lcs175 lcs65 lp mitten150 qt36 qt72"
log() { echo "$(date -u +%FT%TZ) $*" >> $T/logs/tonight.log; }
cd $T
while [ ! -f logs/runner_sc_done ]; do sleep 60; done
log "sc pass done ($(ls results_tcbp_jobs_sc_gpu | grep -c 'json$') results); refitting 21 models"
CORES=192 $PY -m eccentric.decoders.gpu.tcbp_models fit-table tools/matchmaker/data/dataset.csv \
  /home/manosgior/eccentric-validation --force --codes $CODES >> logs/refit.log 2>&1
log "refit exit $? ($(grep -c fitted logs/refit.log) fitted, $(grep -c FAILED logs/refit.log) failed)"
RERUN=$(python3 -c "import json; print(' '.join(map(str, json.load(open('rerun.json')))))")
mkdir -p results/stale_wrong_circuit results_tcbp_jobs_sc_gpu/stale_wrong_circuit
for i in $RERUN; do
  [ -f results/$i.json ] && mv results/$i.json results/stale_wrong_circuit/
  [ -f results_tcbp_jobs_sc_gpu/$i.json ] && mv results_tcbp_jobs_sc_gpu/$i.json results_tcbp_jobs_sc_gpu/stale_wrong_circuit/
done
rm -f logs/runner_sc_done
log "rerunning tcbp_jobs_sc_gpu on $(echo $RERUN | wc -w) circuits"
./runner_sc.sh
log "sc rerun done ($(ls results_tcbp_jobs_sc_gpu | grep -c 'json$') results); GPU pilots of the 39"
cd $P && mkdir -p results/stale_wrong_circuit
for i in $(python3 -c "import json; print(' '.join(str(b[0]) for b in json.load(open('$T/mismatched.json'))))"); do
  [ -f results/$i.jsonl ] && mv results/$i.jsonl results/stale_wrong_circuit/
  rm -rf claims/$i prep/$i.pkl
done
rm -f logs/runner_done
./start.sh
while [ ! -f logs/runner_done ]; do sleep 60; done
log "pilots done ($(ls results | grep -c 'jsonl$') circuits); all finished"
