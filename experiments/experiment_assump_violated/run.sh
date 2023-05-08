#!/usr/bin/bash

EXPERIMENTS=("same_errors" "no_pure_children")

# Run experiments in parallel and store pids in array
for exp in ${EXPERIMENTS[*]}; do
    python3 -m experiments.experiment_assump_violated.run_$exp &
    pids[${i}]=$!
done

# Wait for all pids
for pid in ${pids[*]}; do
    wait $pid
done

# Create plots
python3 -m experiments.experiment_assump_violated.create-plots