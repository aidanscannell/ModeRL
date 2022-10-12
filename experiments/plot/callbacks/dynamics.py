#!/usr/bin/env python3
from functools import partial
from typing import List

import matplotlib.pyplot as plt
import palettable
from experiments.plot.callbacks import KearsPlottingCallback
from experiments.plot.utils import create_test_inputs, plot_contf
from moderl.dynamics import ModeRLDynamics
from mosvgpe.custom_types import InputData


plt.style.use("seaborn-paper")
CMAP = palettable.scientific.sequential.Bilbao_15.mpl_colormap


def plot_gating_network_gps(dynamics: ModeRLDynamics, test_inputs: InputData):
    fig = plt.figure()
    gs = fig.add_gridspec(dynamics.mosvgpe.num_experts, 2)
    axs = gs.subplots()
    mean, var = dynamics.mosvgpe.gating_network.predict_h(test_inputs)
    for k in range(dynamics.mosvgpe.num_experts):
        # label = (
        #     r"$\mathbb{E}[h_{"
        #     + str(dynamics.desired_mode + 1)
        #     + r"}(\mathbf{x}) \mid \mathcal{D}_{0:"
        #     # + str(iteration)
        #     + "}]$"
        # )
        plot_contf(axs[k, 0], test_inputs, z=mean[:, k])
        # plot_contf(axs[0], test_inputs, z=mean[:, dynamics.desired_mode], label=label)
        # label = (
        #     r"$\mathbb{V}[h_{"
        #     + str(dynamics.desired_mode + 1)
        #     + r"}(\mathbf{x}) \mid \mathcal{D}_{0:"
        #     # + str(iteration)
        #     + "}]$"
        # )
        plot_contf(axs[k, 1], test_inputs, z=var[:, k])
        # plot_contf(axs[1], test_inputs, z=var[:, dynamics.desired_mode], label=label)
    # axs[-1].legend()
    fig.tight_layout()
    return fig


def plot_mixing_probs(dynamics: ModeRLDynamics, test_inputs: InputData):
    fig = plt.figure()
    gs = fig.add_gridspec(1, dynamics.mosvgpe.num_experts)
    axs = gs.subplots()
    probs = dynamics.mosvgpe.gating_network.predict_mixing_probs(test_inputs)
    for k in range(dynamics.mosvgpe.num_experts):
        plot_contf(axs[k], test_inputs, z=probs[:, k])
    fig.tight_layout()
    return fig


def build_dynamics_plotting_callbacks(
    dynamics: ModeRLDynamics, logging_epoch_freq: int = 100, num_test: int = 100
) -> List[PlottingCallback]:
    test_inputs = create_test_inputs(num_test=num_test)

    callbacks = [
        KearsPlottingCallback(
            partial(
                plot_gating_network_gps, dynamics=dynamics, test_inputs=test_inputs
            ),
            logging_epoch_freq=logging_epoch_freq,
            name="Gating function posterior",
        ),
        KearsPlottingCallback(
            partial(plot_mixing_probs, dynamics=dynamics, test_inputs=test_inputs),
            logging_epoch_freq=logging_epoch_freq,
            name="Mixing probs",
        ),
    ]
    return callbacks
