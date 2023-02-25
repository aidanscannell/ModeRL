# Experiments accompanying paper
I use [hydra](https://hydra.cc/) to configure the experiments and [Weights & Biases](https://wandb.ai/site)
for experiment tracking.

## Install
Install `ModeRL` with the dependencies for running experiments using
```
pip install --editable ".[experiments]"
```

## Configuratin with Hydra
All of the [experiments](./configs/experiment) use the base hydra
config in [./configs/main.yaml](./configs/main.yaml) but override it differently.
The overrides can be seen in [experiments](./configs/experiment).
Alternatively, an experiments config can be viewed with
``` shell
python train.py +experiment=INSERT_EXPERIMENT_NAME --cfg job
```
You can display the base config using
``` shell
python train.py --cfg=job
```
The experiments in [./configs/experiment](./configs/experiment) are as follows:
- greedy_no_constraint
    - Greedy exploitation strategy with no mode constraint
- greedy_with_constraint
    - Greedy exploitation strategy with mode constraint
- joint_gating_function_entropy
    - ModeRL's main strategy which uses the joint entropy of the gating function over a trajectory
- independent_gating_function_entropy
    - Myopic strategy which uses the mean of the gating function's entropy at each state
- bernoulli
    - Uses the entropy of the mode indicator variable which represents aleatoric uncertainty

## Run experiments
To reproduce an experiment,
``` shell
cd /path/to/experiments
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
