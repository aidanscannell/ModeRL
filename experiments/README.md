# Experiments for Mode Constrained Model-Based Reinforcement Learning via Gaussian Processes

## Configuratin with Hydra
We use hydra to configure the experiments. All experiments use the same base config in [./configs/main.yaml]().
You can display the base config using,
``` shell
python train.py --cfg=job
```


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
or
``` shell
python train.py +experiment=[joint_gating_function_entropy,greedy_no_constraint,greedy_with_constraint,independent_gating_function_entropy,bernoulli]
```

An experiments configuration can be viewed with,
``` shell
python train.py +experiment=greedy_no_constraint --cfg job
```

## Create figures

To create the figures for the paper run,
``` shell
python plot/plot_all_figures.py --wandb_dir=triton --saved_runs=saved_runs.yaml --random_seed=42
```
