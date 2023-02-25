# Using git subtrees
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
