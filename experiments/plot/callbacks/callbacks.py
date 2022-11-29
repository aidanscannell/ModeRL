#!/usr/bin/env python3
from functools import partial
from typing import List

import tensorflow as tf
import wandb
from dynamics import plot_gating_network_gps, plot_mixing_probs
from moderl.dynamics import ModeRLDynamics

from ..utils import create_test_inputs


# from typing import Callable
# import matplotlib as mpl
# PlotFn = Callable[[], mpl.figure.Figure]
PlotFn = None


class KerasPlottingCallback(tf.keras.callbacks.Callback):
    def __init__(self, plot_fn: PlotFn, logging_epoch_freq: int = 10, name: str = ""):
        self.plot_fn = plot_fn
        self.logging_epoch_freq = logging_epoch_freq
        self.name = name

    def on_epoch_end(self, epoch: int, logs=None):
        if epoch % self.logging_epoch_freq == 0:
            fig = self.plot_fn()
            wandb.log({self.name: wandb.Image(fig)})
            # wandb.log({self.name: fig})


# class WandBImageCallbackScipy:
#     def __init__(
#         self, plot_fn: PlotFn, logging_epoch_freq: int = 10, name: Optional[str] = ""
#     ):
#         self.plot_fn = plot_fn
#         self.logging_epoch_freq = logging_epoch_freq
#         self.name = name

#     def __call__(self, step, variables, value):
#         if step % self.logging_epoch_freq == 0:
#             fig = self.plot_fn()
#             wandb.log({self.name: wandb.Image(fig)})


def build_dynamics_plotting_callbacks(
    dynamics: ModeRLDynamics, logging_epoch_freq: int = 100, num_test: int = 100
) -> List[KerasPlottingCallback]:
    test_inputs = create_test_inputs(num_test=num_test)

    callbacks = [
        KerasPlottingCallback(
            partial(
                plot_gating_network_gps, dynamics=dynamics, test_inputs=test_inputs
            ),
            logging_epoch_freq=logging_epoch_freq,
            name="Gating function posterior",
        ),
        KerasPlottingCallback(
            partial(plot_mixing_probs, dynamics=dynamics, test_inputs=test_inputs),
            logging_epoch_freq=logging_epoch_freq,
            name="Mixing probs",
        ),
    ]
    return callbacks
