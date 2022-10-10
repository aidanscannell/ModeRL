#!/usr/bin/env sh

for experiment in configs/experiment/*.yaml; do  # loop through experiments
    echo "file=$experiment"
    sbatch run_experiment.slrm +experiment=$experiment
done
