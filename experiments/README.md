# Experiments for Mode Constrained Model-Based Reinforcement Learning via Gaussian Processes
We use hydra to configure the experiments. All experiments use the same

The default configuration is in [./configs/main.yaml]()

## Run experiments
To reproduce the experiments

``` shell
cd /path/to/experiments
python train.py +experiment=greedy_no_constraint
```
All experiment can be run with,
``` shell
python train.py  --multirun '+experiment=glob(*)'
```

An experiments configuration can be viewed with,
``` shell
python train.py +experiment=greedy_no_constraint --cfg job
```

## Create figures

``` shell
python plot/plot_all_figures.py --wandb_dir=triton --saved_runs=saved_runs.yaml --random_seed=42
```
