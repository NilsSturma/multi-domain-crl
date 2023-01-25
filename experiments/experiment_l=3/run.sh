#!/usr/bin/bash

DOMAINS=(2 3)

# Run experiments in parallel and store pids in array
for i in ${DOMAINS[*]}; do
    python3 -m experiments.experiment_l=3.run_ndom=$i &
    pids[${i}]=$!
done

# Wait for all pids
for pid in ${pids[*]}; do
    wait $pid
done

# Create plots
python3 -m experiments.experiment_l=3.create-plots