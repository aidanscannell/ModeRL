#!/usr/bin/env sh

for path in configs/experiment/*.yaml; do  # loop through experiments
	echo "Running: python train.py +experiment=$(basename ${path} .yaml)"
	sbatch run_experiment.slrm "+experiment=$(basename ${path} .yaml) ++controller.exploration_weight=20"
done
