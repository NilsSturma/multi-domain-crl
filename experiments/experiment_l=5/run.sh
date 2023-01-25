#!/usr/bin/bash

DOMAINS=(2 3 4)

# Run experiments in parallel and store pids in array
for i in ${DOMAINS[*]}; do
    python3 -m experiments.experiment_l=5.run_ndom=$i &
    pids[${i}]=$!
done

# Wait for all pids
for pid in ${pids[*]}; do
    wait $pid
done

# Create plots
python3 -m experiments.experiment_l=5.create-plots