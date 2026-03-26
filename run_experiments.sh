#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

echo "Running size experiments..."
python3 main.py experiments_conf/size.yaml

echo "Running connectivity experiments..."
python3 main.py experiments_conf/connectivity.yaml

echo "Running variance experiments..."
python3 main.py experiments_conf/variance.yaml

echo "Running technologies experiments..."
python3 main.py experiments_conf/technologies.yaml

echo "Running DQC..."
python3 get_dqc.py experiments_conf/dqc.yaml

echo "Running routing stats..."
python3 get_routing_stats.py

echo "Running translate stats..."
python3 get_translate_stats.py

echo "Running accumulated error stats..."
python3 get_accumulated_err.py experiments_conf/accumulated.yaml

echo "Running decoder..."
python3 get_decoder.py experiments_conf/decoder.yaml

echo "Running chromobius decoder..."
python3 get_chromobius.py experiments_conf/decoder_chromobius.yaml

echo "Running program stats..."
python3 get_program_stats.py experiments_conf/program_stats.yaml

echo "Generating plots..."
python3 plots/plots.py

echo "Running IBM demo..."
python3 ibm_demo.py

echo "All experiments and plots completed successfully!"
