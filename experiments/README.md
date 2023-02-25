# Experiments accompanying paper
All experiments are configured using [hydra](https://hydra.cc/) and monitored using [Weights & Biases](https://wandb.ai/site).

## Configuratin with Hydra
All [experiments](./configs/experiment) use the base hydra
config in [./configs/main.yaml](./configs/main.yaml) but override it differently.
The overrides can be seen in [./configs/experiment](./configs/experiment).
An experiment's config can be viewed with:
``` shell
python train.py +experiment=INSERT_EXPERIMENT_NAME --cfg job
```
The base config can be displayed with:
``` shell
python train.py --cfg=job
```
The experiments in [./configs/experiment](./configs/experiment) are as follows:
- greedy_no_constraint
    - Greedy exploitation strategy with no mode constraint
- greedy_with_constraint
    - Greedy exploitation strategy with mode constraint
- moderl
    - ModeRL's main strategy which uses the joint entropy of the gating function over a trajectory
- myopic_ablation
    - Myopic strategy which uses the mean of the gating function's entropy at each state
- aleatoric_unc_ablation
    - Uses the entropy of the mode indicator variable which represents aleatoric uncertainty
- constraint_schedule
    - Uses an exponentially decaying schedule on the constraint level $\delta$ to tighten the constraint during training
- compare_constraint_levels
    - Runs a sweep over constraint levels, i.e. it runs separate experiments for $\delta \in \{0.5, 0.4, 0.3, 0.2, 0.1\}$

## Install
Create a virtual environment:
```
cd /path/to/moderl
python -m venv moderl-venv
source moderl-venv/bin/activate
```
Install `ModeRL` in editable mode with dependencies needed for experiments:
```
pip install -e ".[experiments]"
```

## Running experiments
An experiment can be run with,
``` shell
python train.py +experiment=INSERT_EXPERIMENT_NAME
```
All experiment can be run with,
``` shell
python train.py  --multirun '+experiment=glob(*)'
```
or
``` shell
python train.py --multirun +experiment=moderl,greedy_no_constraint,greedy_with_constraint,myopic_ablation,aleatoric_unc_ablation
python train.py --multirun +experiment=constraint_schedule ++constraint_schedule.decay_episodes=10.0,15.0,20.0
```
Figure 1
``` shell
python train.py --multirun +experiment=moderl
```
Figure 2
``` shell
python train.py --multirun +experiment=constraint_schedule ++controller.mode_satisfaction_prob=0.72
```
Figure 3 - greedy  plots (left)
``` shell
python train.py --multirun +experiment=greedy_no_constraint,greedy_with_constraint
```
Figure 3 - myopic ablation plots (right)
``` shell
python train.py --multirun +experiment=myopic_ablation
```
Figure 5
``` shell
python train.py --multirun +experiment=aleatoric_unc_ablation
```
Figure 6
``` shell
python train.py --multirun +experiment=compare_constraint_levels ++training.random_seed=1,42,69,100,50
```


## Create figures
To create the figures for the paper run,
``` shell
python plot/plot_all_figures.py --wandb_dir=wandb --saved_runs=saved_runs.yaml
```




## Running experiments on Triton (Aalto's cluster)
### Setup the environment
Clone the repo with
``` shell
git clone https://github.com/aidanscannell/ModeRL.git ~/python-projects/moderl
```
Create a virtual environment
``` shell
module load py-virtualenv
python -m venv moderl-venv
```
Install dependencies with
``` shell
cd /path/to/ModeRL/
source moderl-venv/bin/activate
pip install -e ".[experiments]"
```
### Run multiple experiments in parallel whilst using hydra's sweep
``` shell
python train.py --multirun ++controller.mode_satisfaction_prob=0.6,0.7,0.8,0.9 ++training.random_seed=42,1,69,22,4
```

constraint_schedule
### Copy results
Copy wandb results from triton with,
``` shell
rsync -avz -e  "ssh" scannea1@triton.aalto.fi:/home/scannea1/python-projects/aistats-2023/ModeRL/experiments/wandb ./wandb
```

An experiment can be run interactively with something like,
``` shell
srun --mem-per-cpu=500M --cpus-per-task=4 --time=0:10:00 python train.py
```
