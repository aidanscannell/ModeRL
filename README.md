# ModeRL - Mode Remaining Exploration for Model-Based Reinforcement Learning

`ModeRL` is a model-based reinforcement learning method for exploring environments with multimodal transition dynamics.
In particular, it provides probabilistic guarantees on remaining in a desired dynamics mode. 
For example, this may be desirable behaviour if some of the dynamics modes are believed to be inoperable.
`ModeRL` learns representations of multimodal dynamical systems using the Mixture of Sparse Variational Gaussian Process Experts model from [mosvgpe](https://github.com/aidanscannell/mosvgpe).
It then make decisions under the uncertainty of the learned dynamics model.

## Using git subtrees
I use git subtrees to manage dependencies on my own packages. In particular, my Mixture of Sparse Variational Gaussian 
Process Experts package [mosvgpe](https://github.com/aidanscannell/mosvgpe) and my simulation environments 
package [simenvs](https://github.com/aidanscannell/simenvs).

If I make changes to [https://github.com/aidanscannell/mosvgpe](https://github.com/aidanscannell/mosvgpe) I can pull them using,
```
git subtree pull --prefix=subtrees/mosvgpe mosvgpe-subtree master
```
And when I make changes to =mosvgpe= in =moderl= I can push the changes back
to [https://github.com/aidanscannell/mosvgpe](https://github.com/aidanscannell/mosvgpe) using,
```
git subtree push --prefix=subtrees/mosvgpe mosvgpe-subtree /branch/to/push/to
```
For example,
```
git subtree push --prefix=subtrees/mosvgpe mosvgpe-subtree aidanscannell/push-from-moderl
```
