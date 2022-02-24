#!/usr/bin/env python3
from typing import Callable, Optional

import matplotlib
import tensorflow as tf
from mogpe.keras.callbacks.tensorboard import plot_to_image

PlotFn = Callable[[], matplotlib.figure.Figure]


class TensorboardImageCallbackScipy:
    def __init__(
        self,
        plot_fn: PlotFn,
        logging_epoch_freq: int = 10,
        log_dir: Optional[str] = "./logs",
        name: Optional[str] = "",
    ):
        self.plot_fn = plot_fn
        self.logging_epoch_freq = logging_epoch_freq
        self.name = name

        self.file_writer = tf.summary.create_file_writer(log_dir)

    def __call__(self, step, variables, value):
        if step % self.logging_epoch_freq == 0:
            figure = self.plot_fn()
            with self.file_writer.as_default():
                tf.summary.image(self.name, plot_to_image(figure), step=step)
