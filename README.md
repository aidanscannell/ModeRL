# ModeRL: Mode-constrained model-based reinforcement learning
<p align="center">
    <img src="https://github.com/aidanscannell/moderl/blob/master/experiments/gifs/initial-submission/moderl-exploration.gif" alt="ModeRL">
</p>

<!-- <p align="center"> -->
<!--     <embed src="https://github.com/aidanscannell/moderl/blob/master/experiments/figures/initial_submission/joint_gating_four_iterations_in_row.pdf" alt="ModeRL" /> -->
<!--     <\!-- <embed src="./experiments/figures/initial_submission/joint_gating_four_iterations_in_row.pdf" width="800px" height="2100px" alt="ModeRL" /> -\-> -->
<!-- </p> -->
<!-- <\!-- <img align="middle" src="./experiments/figures/initial_submission/moderl_four_iterations_in_row.pdf" width="666" /> -\-> -->
<!-- <p align="center"> -->
<!-- <img align="middle" src="./experiments/figures/initial_submission/joint_gating_four_iterations_in_row.pdf" width="666" /> -->
<!-- </p> -->

`ModeRL` is a model-based reinforcement learning method that attempts to constrain learning to a single dynamics.
It simultaneously learns and enforces the mode constraint by
learning a representation of the dynamics using the Mixtures of Sparse Variational Gaussian Process Experts
method from [mosvgpe](https://github.com/aidanscannell/mosvgpe).
It then makes decisions under the uncertainty of the learned dynamics model to provide probabilistic guarantees
of remaining in the desired dynamics mode.

## Usage
- See [experiments](./experiments) for details on running the experiments in our AISTATS paper.
- See the notebook in [examples](./examples) for how to configure and run `ModeRL`.


## Running and plotting
Create a virtual environment:
```
cd /path/to/moderl
python -m venv moderl-venv
source moderl-venv/bin/activate
```
Install `ModeRL` in editable mode with dependencies needed for experiments:
```
cd /path/to/moderl
python -m venv moderl-venv
source moderl-venv/bin/activate
pip install -e ".[experiments]"
```
Run experiments:
``` shell
cd /path/to/experiments
python train.py +experiment=INSERT_EXPERIMENT_NAME
```

## Using git subtrees
I use git subtrees to manage dependencies on my own packages. In particular, my Mixtures of Sparse Variational Gaussian
Process Experts package [mosvgpe](https://github.com/aidanscannell/mosvgpe) and my simulation environments
package [simenvs](https://github.com/aidanscannell/simenvs).

If I make changes to [https://github.com/aidanscannell/mosvgpe](https://github.com/aidanscannell/mosvgpe) I can pull them using,
```
git subtree pull --prefix=subtrees/mosvgpe mosvgpe-subtree master
```
And when I make changes to `mosvgpe` in `moderl` I can push the changes back
to [https://github.com/aidanscannell/mosvgpe](https://github.com/aidanscannell/mosvgpe) using,
```
git subtree push --prefix=subtrees/mosvgpe mosvgpe-subtree /branch/to/push/to
```
For example,
```
git subtree push --prefix=subtrees/mosvgpe mosvgpe-subtree aidanscannell/push-from-moderl
```

### Citation
```bibtex
@proceedings{scannell2023,
    title={Mode-constrained Model-based Reinforcement Learning via Gaussian Processes},
    author={Scannell, Aidan and Ek, Carl Henrik and Richards, Arthur},
    booktitle = {International {{Conference}} on {{Artificial Intelligence}} and {{Statistics}}},
    year={2023}
}
```
