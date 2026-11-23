#!/usr/bin/env bash
# After the TCBP models are fitted (tcbp_models fit-table, logs/fit_table.log), run the slice (runner.sh), detached
HERE=/home/manosgior/eccentric-validation/gpu_dev/tcbp_slice
cd $HERE && mkdir -p logs
(setsid nohup bash -c "while pgrep -f '[t]cbp_models fit-table' >/dev/null; do sleep 30; done; $HERE/runner.sh" > logs/start.log 2>&1 < /dev/null &)
