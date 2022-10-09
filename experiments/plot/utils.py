#!/usr/bin/env python3
import os
from functools import partial
from typing import Callable, List

import gpflow as gpf
import hydra
import keras
import matplotlib
import matplotlib.pyplot as plt

# import matplotlib.pyplot as plt
import numpy as np
import omegaconf
import palettable
import simenvs
import tensorflow as tf
import tikzplotlib
from matplotlib import patches
from moderl.controllers import ControllerInterface
from moderl.controllers.explorative_controller import ExplorativeController
from moderl.dynamics import ModeRLDynamics
from moderl.dynamics.dynamics import ModeRLDynamics
from mosvgpe.custom_types import Dataset, InputData
from mosvgpe.mixture_of_experts import MixtureOfSVGPExperts
from mpl_toolkits.axes_grid1 import make_axes_locatable

import wandb

plt.style.use("seaborn-paper")
CMAP = palettable.scientific.sequential.Bilbao_15.mpl_colormap


PlotFn = Callable[[], matplotlib.figure.Figure]

colors = ["m", "c", "y"]


def create_test_inputs(num_test: int = 400):
    sqrtN = int(np.sqrt(num_test))
    xx = np.linspace(-4, 3, sqrtN)
    yy = np.linspace(-4, 4, sqrtN)
    xx, yy = np.meshgrid(xx, yy)
    test_inputs = np.column_stack([xx.reshape(-1), yy.reshape(-1)])
    zeros = np.zeros((num_test, 2))
    test_inputs = np.concatenate([test_inputs, zeros], -1)
    return test_inputs


class PlottingCallback(tf.keras.callbacks.Callback):
    def __init__(self, plot_fn: PlotFn, logging_epoch_freq: int = 10, name: str = ""):
        self.plot_fn = plot_fn
        self.logging_epoch_freq = logging_epoch_freq
        self.name = name

    def on_epoch_end(self, epoch: int, logs=None):
        if epoch % self.logging_epoch_freq == 0:
            fig = self.plot_fn()
            wandb.log({self.name: wandb.Image(fig)})
            # wandb.log({self.name: fig})


def plot_desired_mixing_prob(ax, dynamics: ModeRLDynamics, test_inputs: InputData):
    probs = dynamics.mosvgpe.gating_network.predict_mixing_probs(test_inputs)
    return plot_contf(ax, test_inputs, z=probs[:, dynamics.desired_mode])


def plot_gating_function_mean(ax, dynamics: ModeRLDynamics, test_inputs: InputData):
    mean, _ = dynamics.mosvgpe.gating_network.predict_h(test_inputs)
    label = (
        "$\mathbb{E}[h_{"
        + str(dynamics.desired_mode + 1)
        + "}(\mathbf{x}) \mid \mathcal{D}_{0:"
        # + str(iteration)
        + "}]$"
    )
    return plot_contf(ax, test_inputs, z=mean[:, dynamics.desired_mode])


def plot_gating_function_variance(ax, dynamics: ModeRLDynamics, test_inputs: InputData):
    _, var = dynamics.mosvgpe.gating_network.predict_h(test_inputs)
    label = (
        "$\mathbb{V}[h_{"
        + str(dynamics.desired_mode + 1)
        + "}(\mathbf{x}) \mid \mathcal{D}_{0:"
        # + str(iteration)
        + "}]$"
    )
    return plot_contf(ax, test_inputs, z=var[:, dynamics.desired_mode])


def plot_mode_satisfaction_prob(
    ax,
    dynamics: ModeRLDynamics,
    test_inputs: InputData,
    mode_satisfaction_prob: float,
):
    mixing_probs = dynamics.mosvgpe.gating_network.predict_mixing_probs(test_inputs)
    CS = ax.tricontour(
        test_inputs[:, 0],
        test_inputs[:, 1],
        mixing_probs[:, dynamics.desired_mode].numpy(),
        [mode_satisfaction_prob],
    )
    ax.clabel(CS, inline=True, fontsize=10)


def plot_contf(ax, test_inputs, z, levels=None, cmap=CMAP):
    try:
        contf = ax.tricontourf(
            test_inputs[:, 0],
            test_inputs[:, 1],
            z,
            # 100,
            levels=levels,
            cmap=cmap,
        )
    except ValueError:
        # TODO check this works
        contf = ax.tricontourf(
            test_inputs[:, 0],
            test_inputs[:, 1],
            np.ones(z.shape),
            # 100,
            levels=levels,
            cmap=cmap,
        )
    return contf


# def cbar(fig, ax, contf):
#     if isinstance(ax, np.ndarray):
#         divider = make_axes_locatable(ax[0])
#     else:
#         divider = make_axes_locatable(ax)
#     cax = divider.append_axes("top", size="5%", pad=0.05)
#     cbar = fig.colorbar(
#         contf,
#         ax=ax,
#         use_gridspec=True,
#         cax=cax,
#         # format="%0.2f",
#         orientation="horizontal",
#     )
#     # cbar.ax.locator_params(nbins=9)

#     # cax.ticklabel_format(style="sci", scilimits=(0, 3))
#     cax.xaxis.set_ticks_position("top")
#     cax.xaxis.set_label_position("top")
#     return cbar


def plot_data_and_traj_over_desired_mixing_prob(
    ax,
    dynamics: ModeRLDynamics,
    controller: ExplorativeController,
    test_inputs: InputData,
):
    plot_desired_mixing_prob(ax, dynamics=dynamics, test_inputs=test_inputs)
    plot_data_over_ax(ax, x=dynamics.dataset[0][:, 0], y=dynamics.dataset[0][:, 1])
    plot_mode_satisfaction_prob(
        ax,
        dynamics=dynamics,
        test_inputs=test_inputs,
        mode_satisfaction_prob=controller.mode_satisfaction_prob,
    )


def plot_data_over_ax(ax, x, y):
    ax.scatter(
        x,
        y,
        marker="x",
        color="b",
        linewidth=0.5,
        alpha=0.5,
        label="Observations",
    )
