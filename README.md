# ModeRL: Mode-constrained Model-based Reinforcement Learning via Gaussian Processes
This repo contains the code and source docs for our paper:
<table>
    <tr>
        <td>
            <strong>Mode-constrained Model-based Reinforcement Learning via Gaussian Processes</strong><br>
            <i>Proceedings of the 26th International Conference on Artificial Intelligence and Statistics (AISTATS) 2023</i><br>
            Aidan Scannell, Carl Henrik Ek, Arthur Richards <br>
            <a href="https://www.aidanscannell.com/publication/mode-constrained-mbrl/paper.pdf"><img alt="Paper" src="https://img.shields.io/badge/-Paper-gray"></a>
            <!-- <a href="https://www.aidanscannell.com/publication/mode-constrained-mbrl/"><img alt="Website" src="https://img.shields.io/badge/-Website-gray" ></a></br> -->
        </td></br>
    </tr>
    <tr>
        <td>
        Model-based reinforcement learning (RL) algorithms do not typically consider environments with multiple dynamic modes, where it is beneficial to avoid inoperable or undesirable modes. We present a model-based RL algorithm that constrains training to a single dynamic mode with high probability. This is a difficult problem because the mode constraint is a hidden variable associated with the environment’s dynamics. As such, it is 1) unknown a priori and 2) we do not observe its output from the environment, so cannot learn it with supervised learning. We present a nonparametric dynamic model which learns the mode constraint alongside the dynamic modes. Importantly, it learns latent structure that our planning scheme leverages to 1) enforce the mode constraint with high probability, and 2) escape local optima induced by the mode constraint. We validate our method by showing that it can solve a simulated quadcopter navigation task whilst providing a level of constraint satisfaction both during and after training.
        </td>
    </tr>
</table>

<p align="center">
    <img src="https://github.com/aidanscannell/moderl/blob/master/experiments/gifs/initial-submission/moderl-exploration.gif" alt="ModeRL">
</p>


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

## Running and plotting
See [experiments/](./experiments) for detailed instructions on running all of the experiments in the paper.
As an example, the `ModeRL` experiment with a schedule that tightens the constraint level during training can be run with:
``` shell
cd ./experiments
python train.py +experiment=constraint_schedule
```
See the [example notebook](./examples/quadcopter-navigation-via-mode-constrained-mbrl.ipynb) to see how to use `ModeRL` in practice.

## Citation
```bibtex
@proceedings{scannell2023moderl,
    title={Mode-constrained Model-based Reinforcement Learning via Gaussian Processes},
    author={Scannell, Aidan and Ek, Carl Henrik and Richards, Arthur},
    booktitle = {International {{Conference}} on {{Artificial Intelligence}} and {{Statistics}}},
    year={2023}
}
```
