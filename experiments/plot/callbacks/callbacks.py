#!/usr/bin/env python3
from typing import Callable

import matplotlib as mpl
import tensorflow as tf
import wandb


PlotFn = Callable[[], mpl.figure.Figure]


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
