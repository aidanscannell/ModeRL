# Experiments for Mode Constrained Model-Based Reinforcement Learning via Gaussian Processes

## Configuratin with Hydra
I use hydra to configure the experiments. All of the [experiments](./configs/experiment) use the base
config in [](./configs/main.yaml) but override it differently.
The overrides can be seen in [experiments](./configs/experiment) or an experiments config can be viewed with
``` shell
python train.py +experiment=INSERT_EXPERIMENT_NAME --cfg job
```
You can display the base config using
``` shell
python train.py --cfg=job
```
The experiments in [](./configs/experiment) are as follows:
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
python train.py +experiment=[joint_gating_function_entropy,greedy_no_constraint,greedy_with_constraint,independent_gating_function_entropy,bernoulli]
```


## Create figures
To create the figures for the paper run,
``` shell
python plot/plot_all_figures.py --wandb_dir=triton --saved_runs=saved_runs.yaml --random_seed=42
```

## Running experiments on Triton (Aalto's cluster)
### Setup the environment
Clone the repo with
``` shell
cd ~/python-projects
git clone https://github.com/aidanscannell/ModeRL.git
```
Create a virtual environment
``` shell
module load py-virtualenv
python -m venv moderl-venv
```
Install dependencies with
``` shell
cd /path/to/ModeRL/
pip install -e ".[experiments]"
```
### Run experiments
Run all experiments with
``` shell
chmod +x run_all_experiments.sh
./run_all_experiments.sh
```
Alternatively, run a single experiment with,
``` shell
sbatch run_experiment.slrm INSERT_EXPERIMENT_NAME
```

### Copy results
Copy wandb results from triton with,
``` shell
rsync -e "ssh" -avz scannea1@triton.aalto.fi:/home/scannea1/python-projects/moderl/experiments/wandb/* ./
```

An experiment can be run interactively with something like,
``` shell
srun --mem-per-cpu=500M --cpus-per-task=4 --time=0:10:00 python train.py
```
