# Experiments accompanying paper
All experiments are configured using [hydra](https://hydra.cc/) and monitored using [Weights & Biases](https://wandb.ai/site).

## Experiment configuratin with Hydra
All [experiments](./configs/experiment) use the base hydra
config in [./configs/main.yaml](./configs/main.yaml) but override it differently.
The overrides can be seen in [./configs/experiment](./configs/experiment).
An experiment's config can be viewed with:
``` shell
python train.py +experiment=INSERT_EXPERIMENT_NAME --cfg job
```
where `INSERT_EXPERIMENT_NAME` is the filename of an experiment's `yaml` config in [./configs/experiment](./configs/experiment).
The base config can be displayed with:
``` shell
python train.py --cfg=job
```

<table class=".table" style="width:100%">
  <thead>
  <tr>
    <td>Experiment</td>
    <td>Description</td>
    </tr>
  </thead>
  <tbody>
  <tr>
    <td style="width:10%">
<img src="https://github.com/aidanscannell/moderl/blob/master/experiments/gifs/initial-submission/greedy-no-constraint.gif" alt="<b>Greedy exploitation WITHOUT mode constraint</b>"> </td>
    <td style="width:10%">
     `greedy_no_constraint` - We are not able to solve our δ-mode-constrained navigation problem with the greedy exploitation strategy because it leaves the desired dynamics mode.</td>
  </tr>
  <tr>
    <td style="width:10%">
<img src="https://github.com/aidanscannell/moderl/blob/master/experiments/gifs/initial-submission/greedy-with-constraint.gif" alt="<b>Greedy exploitation WITH mode constraint</b>"> </td>
    <td style="width:10%">
    `greedy_with_constraint` - Adding the δ-mode-constraint to the greedy exploitation strategy is still not able to solve our δ-mode-constrained navigation problem. This is because the optimisation gets stuck at a local optima induced by the constraint.
     </td>
  </tr>
  <tr>
    <td style="width:10%">
<img src="https://github.com/aidanscannell/moderl/blob/master/experiments/gifs/initial-submission/moderl-exploration.gif" alt="<b>ModeRL (ours)</b>"> </td>
    <td style="width:10%">
    `moderl` - Our strategy successfully solves our δ-mode-constrained navigation problem by augmenting the greedy exploitation objective with an intrinsic motivation term. Our intrinsic motivation uses the epistmic uncertainty associated with the learned mode constraint to escape local optima induced by the constraint.
     </td>
  </tr>
  <tr>
    <td style="width:10%">
<img src="https://github.com/aidanscannell/moderl/blob/master/experiments/gifs/initial-submission/aleatoric-uncertainty.gif" alt="<b>Aleatoric uncertainty (ablation)</b>"> </td>
    <td style="width:10%">
`aleatoric_unc_ablation` - Here we show the importance of using only the epistemic uncertainty for exploration. This experiment augmented the greedy objective with the entropy of the mode indicator variable. It cannot escape the local optimum induced by the mode constraint because the mode indicator variable's entropy is <b>always</b> high at the mode boundary. This motivated formulating a dynamics model which can disentangle the sources of uncertainty in the mode constraint.
     </td>
  </tr>
  <tr>
    <td style="width:10%">
<img src="https://github.com/aidanscannell/moderl/blob/master/experiments/gifs/initial-submission/myopic-moderl.gif" alt="<b>Myopic intrinsic exploration (ablation)</b>"> </td>
    <td style="width:10%">
    `myopic_ablation` - Finally, we motivate why our intrinsic motivatin term considers the joint entroy over a trajectory, instead of summing the entropy at each time step (as is often seen in the literature). This experiment formulated the intrinsic motivation term as the sum of the gating function entropy at each time step. That is, it assumed each time step is independent and did not consider the information gain over an entire trajectory, i.e. the exploration is myopic (aka shortsighted).
     </td>
  </tr>
  </tbody>
</table>



<!-- The experiments in [./configs/experiment](./configs/experiment) are as follows: -->
<!-- - greedy_no_constraint -->
<!--     - Greedy exploitation strategy with no mode constraint -->
<!-- - greedy_with_constraint -->
<!--     - Greedy exploitation strategy with mode constraint -->
<!-- - moderl -->
<!--     - ModeRL's main strategy which uses the joint entropy of the gating function over a trajectory -->
<!-- - myopic_ablation -->
<!--     - Myopic strategy which uses the mean of the gating function's entropy at each state -->
<!-- - aleatoric_unc_ablation -->
<!--     - Uses the entropy of the mode indicator variable which represents aleatoric uncertainty -->
<!-- - constraint_schedule -->
<!--     - Uses an exponentially decaying schedule on the constraint level $\delta$ to tighten the constraint during training -->
<!-- - compare_constraint_levels -->
<!--     - Runs a sweep over constraint levels, i.e. it runs separate experiments for $\delta \in \{0.5, 0.4, 0.3, 0.2, 0.1\}$ -->

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
