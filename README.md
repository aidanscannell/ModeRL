# ModeRL - Mode constrained model-based Reinforcement Learning
`ModeRL` is a model-based reinforcement learning method that constrains learning to a single dynamics.
It simultaneously learns and enforces the mode constraint by
learning a representation of the dynamics using the Mixture of Sparse Variational Gaussian Process Experts model from [mosvgpe](https://github.com/aidanscannell/mosvgpe).
It then makes decisions under the uncertainty of the learned dynamics model to provide probabilistic guarantees
of remaining in the desired dynamics mode.

## Usage
See the notebook in [examples](./examples) for how to configure and run `ModeRL`.
See the [experiments](./experiments) for how to configure and run `ModeRL`.

## Install
Install `ModeRL` in editable mode using
```
pip install --editable .
```
To install with the developer requirements use
```
pip install --editable ".[experiments]"
```
Or to install with the developer requirements use
```
pip install --editable ".[dev]"
```


## Using git subtrees
I use git subtrees to manage dependencies on my own packages. In particular, my Mixture of Sparse Variational Gaussian
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
