#!/bin/bash

# Sweep parameters
MANIFOLD_STEPS=(1 5 10 25 50 100)
ADMM_RHO=(None 2.0 4.0 8.0 16.0)

# Number of GPUs available
NUM_GPUS=2

# Create a queue of experiments
EXPERIMENTS=()
for steps in "${MANIFOLD_STEPS[@]}"; do
    for rho in "${ADMM_RHO[@]}"; do
        EXPERIMENTS+=("$steps $rho")
    done
done

echo "Total experiments: ${#EXPERIMENTS[@]}"

# Function to run a single experiment
run_experiment() {
    local steps=$1
    local rho=$2
    local device=$3

    echo "Starting experiment: manifold_steps=$steps, admm_rho=$rho on $device"

    if [ "$rho" = "None" ]; then
        uv run src/main.py --manifold_steps "$steps" --device "$device"
    else
        uv run src/main.py --manifold_steps "$steps" --admm_rho "$rho" --device "$device"
    fi

    echo "Completed experiment: manifold_steps=$steps, admm_rho=$rho on $device"
}

# Export function so it's available to parallel processes
export -f run_experiment

# Run experiments in parallel using background processes
index=0
for exp in "${EXPERIMENTS[@]}"; do
    read -r steps rho <<< "$exp"

    # Assign GPU based on current index
    gpu_id=$((index % NUM_GPUS))
    device="cuda:$gpu_id"

    # Wait for a slot to be available
    while [ $(jobs -r | wc -l) -ge $NUM_GPUS ]; do
        sleep 1
    done

    # Run experiment in background
    run_experiment "$steps" "$rho" "$device" &

    ((index++))
done

# Wait for all background jobs to complete
wait

echo "All experiments completed!"
