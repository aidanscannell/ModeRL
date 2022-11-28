#!/usr/bin/env python3
import matplotlib.pyplot as plt
from moderl.dynamics import ModeRLDynamics
from mosvgpe.custom_types import InputData

from ..utils import plot_contf


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
