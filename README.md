
# Table of Contents



*Disclaimer: This is unfinished research code accompanying my PhD.*

`ModeOpt` is a package for learning and controlling unknown, or partially unknown, multimodal dynamical systems.
In particular, it is concerned with methods for learning and control that attempt to remain in a given desired dynamics
mode. For example, if some of the dynamics modes are believed to be unoperatable.
`ModeOpt` learns representations of multimodal dynamical systems using the Mixture of Gaussian Process Experts model from [mogpe](https://github.com/aidanscannell/mogpe).
It then deploys multiple control strategies (trajectory optimisation algorithms) that make decisions
under the uncertainty of the learned dynamics model.

`ModeOpt` consists of trajectory optimisers with two main goals:

1.  Find trajectories between a start and end state that remain in a given dynamics mode and attempt to avoid regions of the dynamics with high epistemic uncertainty.
2.  Find trajectories that guide exploration of the state-control space whilst remaining in a given desired dynamics mode.

