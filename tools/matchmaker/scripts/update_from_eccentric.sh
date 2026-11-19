#!/usr/bin/env bash
# Bring Matchmaker up to date with ECCentric's sweep table (in this repository): extract the features of
# new circuits on the cluster, rebuild the training table, retrain, re-evaluate, and commit the feature store,
# the packaged model and reports/ if anything changed. It does not push.
#
#   tools/matchmaker/scripts/update_from_eccentric.sh
#
# Settings (environment): MM_HOST (cluster
# host with the run directories, default ian), MM_BASE (directory of ECCentric's runs there, default
# ~/eccentric-validation), MM_PROCESSES (default 8), MM_REMOTE_PYTHON (how to run Python with stim on the
# cluster, from $MM_BASE/repo; default ECCentric's nix-shell and venv).
# MM_PYTHON: the local Python with Matchmaker installed (default: python).
set -euo pipefail
cd "$(dirname "$0")/.."
ECCENTRIC=../..
HOST=${MM_HOST:-ian}
BASE=${MM_BASE:-eccentric-validation}
PROCESSES=${MM_PROCESSES:-8}
REMOTE_PYTHON=${MM_REMOTE_PYTHON:-'nix-shell .nix_env/shell.nix --run "../venv/bin/python -W ignore $*"'}
PY=${MM_PYTHON:-python}
RESULTS="$ECCENTRIC/experiment_results/Sweep/results.csv"
COMMIT=$(git -C "$ECCENTRIC" rev-parse --short HEAD)

echo "== features of new circuits on $HOST"
rsync -a --delete --exclude .venv --exclude data --exclude data_interim ./ "$HOST:$BASE/qec-matchmaker/"
scp -q "$RESULTS" "$HOST:$BASE/qec-matchmaker/results.csv"
[ -f data/features.jsonl.gz ] && gunzip -c data/features.jsonl.gz > data/features.jsonl && \
  scp -q data/features.jsonl "$HOST:$BASE/matchmaker_features.jsonl.local" || true
ssh "$HOST" "cd $BASE && { [ -f matchmaker_features.jsonl ] || cp matchmaker_features.jsonl.local matchmaker_features.jsonl 2>/dev/null || true; } && cd repo && \
  export PYTHONPATH=\$HOME/$BASE/qec-matchmaker OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 && \
  set -- -m qec_matchmaker.cli extract \$(ls -d \$HOME/$BASE/sweep_* \$HOME/$BASE/sweep[0-9]* 2>/dev/null | sort -u) \
    --store \$HOME/$BASE/matchmaker_features.jsonl --processes $PROCESSES --keys-from \$HOME/$BASE/qec-matchmaker/results.csv && \
  $REMOTE_PYTHON"
mkdir -p data
scp -q "$HOST:$BASE/matchmaker_features.jsonl" data/features.jsonl
gzip -9 -c data/features.jsonl > data/features.jsonl.gz && rm data/features.jsonl

echo "== training table, models, report (ECCentric $COMMIT)"
$PY -m qec_matchmaker.cli ingest --results "$RESULTS" --store data/features.jsonl.gz --out data --eccentric-commit "$COMMIT"
$PY -m qec_matchmaker.cli train --data data
$PY -m qec_matchmaker.cli evaluate --data data --out reports > /dev/null

git add data/features.jsonl.gz qec_matchmaker/model reports
if git diff --cached --quiet; then
    echo "== no change"
    exit 0
fi
ROWS=$($PY -c "import json; m = json.load(open('data/MANIFEST.json')); print(f\"{m['rows']} rows, {m['circuits']} circuits\")")
git commit -q -m "Matchmaker: retrained on the sweep table of $COMMIT ($ROWS)"
echo "== committed: $(git log --oneline -1)"
