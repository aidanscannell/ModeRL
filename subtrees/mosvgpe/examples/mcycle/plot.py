#!/usr/bin/env python3
from typing import Callable, List
from functools import partial

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from gpflow.inducing_variables import MultioutputInducingVariables
from mosvgpe.custom_types import InputData
from mosvgpe.mixture_of_experts import MixtureOfSVGPExperts

import wandb

PlotFn = Callable[[], matplotlib.figure.Figure]

colors = ["m", "c", "y"]


# class PlottingCallback(tf.keras.callbacks.Callback):
#     def __init__(self, plot_fn: PlotFn, logging_epoch_freq: int = 10, name: str = ""):
#         self.plot_fn = plot_fn
#         self.logging_epoch_freq = logging_epoch_freq
#         self.name = name

#     def on_epoch_end(self, epoch: int, logs=None):
#         if epoch % self.logging_epoch_freq == 0:
#             fig = self.plot_fn()
#             wandb.log({self.name: wandb.Image(fig)})
#             # wandb.log({self.name: fig})


# def plot_experts_gps(model: MixtureOfSVGPExperts, test_inputs: InputData):
#     fig = plt.figure()
#     gs = fig.add_gridspec(1, 1)
#     ax = gs.subplots()
#     for k, expert in enumerate(model.experts_list):
#         mean, var = expert.predict_f(test_inputs)
#         # f_samples = expert.predict_f_samples(test_inputs, 5)
#         Z = expert.gp.inducing_variable.Z
#         # for f_sample in f_samples:
#         #     axs[row].plot(self.test_inputs, f_sample, alpha=0.3)
#         plot_gp(ax, mean[:, 0], var[:, 0], label="k=" + str(k + 1), color=colors[k])
#         ax.scatter(Z, np.zeros(Z.shape), marker="|", color=colors[k])
#     ax.legend()
#     return fig


# def plot_gating_networks_gps(model: MixtureOfSVGPExperts, test_inputs: InputData):
#     fig = plt.figure()
#     gs = fig.add_gridspec(1, 1)
#     ax = gs.subplots()
#     mean, var = model.gating_network.predict_h(test_inputs)
#     if isinstance(
#         model.gating_network.gp.inducing_variable, MultioutputInducingVariables
#     ):
#         Z = model.gating_network.gp.inducing_variable.inducing_variable.Z
#     else:
#         Z = model.gating_network.gp.inducing_variable.Z
#     for k in range(mean.shape[-1]):
#         plot_gp(ax, mean[:, k], var[:, k], label="k=" + str(k + 1), color=colors[k])
#         ax.scatter(Z, np.zeros(Z.shape), marker="|", color=colors[k])
#     ax.legend()
#     return fig


# def plot_mixing_probs(model: MixtureOfSVGPExperts, test_inputs: InputData):
#     fig = plt.figure()
#     gs = fig.add_gridspec(1, 1)
#     ax = gs.subplots()
#     probs = model.gating_network.predict_mixing_probs(test_inputs)
#     for k in range(model.num_experts):
#         ax.plot(test_inputs, probs[:, k], label="k=" + str(k + 1), color=colors[k])
#     ax.legend()
#     return fig


# def plot_posterior(model: MixtureOfSVGPExperts, test_inputs: InputData):
#     fig = plt.figure()
#     gs = fig.add_gridspec(1, 1)
#     ax = gs.subplots()
#     ax.scatter(X, Y, marker="x", color="k")
#     num_samples = 30
#     test_inputs_broadcast = np.broadcast_to(
#         test_inputs, (num_samples, *test_inputs.shape)
#     )
#     alpha_samples = model.gating_network.predict_categorical_dist(test_inputs).sample(
#         num_samples
#     )  # [S, N]
#     alpha_samples = tf.expand_dims(alpha_samples, -1)  # [S, N, 1]
#     experts_dists = model.predict_experts_dists(test_inputs)
#     y_mean = model.predict_y(test_inputs).mean()
#     ax.plot(test_inputs, y_mean, color="k", lw=2, label="Posterior mean")
#     for k in range(model.num_experts):
#         y_samples = experts_dists[k].sample(num_samples)
#         ax.scatter(
#             test_inputs_broadcast[alpha_samples == k],
#             y_samples[alpha_samples == k],
#             c=colors[k],
#             s=3,
#             alpha=0.8,
#             label="k=" + str(k + 1) + " samples",
#         )
#     ax.legend()
#     return fig


# def plot_gp(ax, mean, var, label="", color="C0", alpha=0.4):
#     ax.scatter(X, Y, marker="x", color="k", alpha=alpha)
#     ax.plot(test_inputs, mean, color=color, lw=2, label=label)
#     ax.fill_between(
#         test_inputs[:, 0],
#         mean - 1.96 * np.sqrt(var),
#         mean + 1.96 * np.sqrt(var),
#         color=color,
#         alpha=0.2,
#     )


def buld_plotting_callbacks(
    model: MixtureOfSVGPExperts, logging_epoch_freq: int = 100
) -> List[PlotFn]:
    test_inputs = np.linspace(
        tf.reduce_min(X) * 1.2, tf.reduce_max(X) * 1.2, 100
    ).reshape(-1, 1)
    # callbacks = [
    #     PlottingCallback(
    #         partial(plot_experts_gps, model=model, test_inputs=test_inputs),
    #         logging_epoch_freq=logging_epoch_freq,
    #         name="Expert GPs",
    #     ),
    #     PlottingCallback(
    #         partial(plot_gating_networks_gps, model=model, test_inputs=test_inputs),
    #         logging_epoch_freq=logging_epoch_freq,
    #         name="Gating Network GPs",
    #     ),
    #     PlottingCallback(
    #         partial(plot_mixing_probs, model=model, test_inputs=test_inputs),
    #         logging_epoch_freq=logging_epoch_freq,
    #         name="Mixing Probs",
    #     ),
    #     PlottingCallback(
    #         partial(plot_posterior, model=model, test_inputs=test_inputs),
    #         logging_epoch_freq=logging_epoch_freq,
    #         name="Posterior",
    #     ),
    # ]
    # return callbacks
