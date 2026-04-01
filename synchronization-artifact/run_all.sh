#!/bin/bash

# Run all the scripts in the artifact
# This script should be run from the root of the artifact, within the Docker container

python3 synchronization_zxxz.py 50000000 128
python3 synchronization_zzxx.py 50000000 128
python3 ideal_synchronization_zzxx.py 30000000 128
python3 ideal_synchronization_zxxz.py 30000000 128
python3 hybrid_sync.py 30000000 128
python3 hybrid_sync_na.py 30000000 128
python3 active_with_rounds.py 30000000 128