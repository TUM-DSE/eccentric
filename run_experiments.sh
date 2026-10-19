#!/bin/bash
set -e

echo "Running size experiments..."
python3 main.py experiments_conf/size.yaml

echo "Running connectivity experiments..."
python3 main.py experiments_conf/connectivity.yaml

echo "Running variance experiments..."
python3 main.py experiments_conf/variance.yaml

echo "Running technology experiments..."
python3 main.py experiments_conf/technologies.yaml

echo "Running DQC experiments..."
python3 get_dqc.py experiments_conf/dqc.yaml

echo "Running routing stats..."
python3 get_routing_stats.py routing_experiments.yaml

echo "Running translation stats..."
python3 get_translate_stats.py translate_experiments.yaml

echo "Running accumulated error experiments..."
python3 get_accumulated_err.py experiments_conf/accumulated.yaml

echo "Running decoder experiments..."
python3 get_decoder.py experiments_conf/decoder.yaml

echo "Running Chromobius decoder experiments..."
python3 get_chromobius.py experiments_conf/decoder_chromobius.yaml

echo "Running program stats..."
python3 get_program_stats.py experiments_conf/program_stats.yaml

echo "Generating plots..."
python3 plots/plots.py

echo "Generating real hardware comparison plot..."
python3 ibm_demo.py

echo "All experiments and plots completed successfully!"
