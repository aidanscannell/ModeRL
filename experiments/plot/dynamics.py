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
import wandb
from experiments.plot.utils import PlottingCallback, create_test_inputs, plot_contf
from matplotlib import patches
from moderl.controllers import ControllerInterface
from moderl.controllers.explorative_controller import ExplorativeController
from moderl.dynamics import ModeRLDynamics
from moderl.dynamics.dynamics import ModeRLDynamics
from mosvgpe.custom_types import Dataset, InputData
from mosvgpe.mixture_of_experts import MixtureOfSVGPExperts
from mpl_toolkits.axes_grid1 import make_axes_locatable

plt.style.use("seaborn-paper")
CMAP = palettable.scientific.sequential.Bilbao_15.mpl_colormap


def plot_gating_network_gps(dynamics: ModeRLDynamics, test_inputs: InputData):
    fig = plt.figure()
    gs = fig.add_gridspec(dynamics.mosvgpe.num_experts, 2)
    axs = gs.subplots()
    mean, var = dynamics.mosvgpe.gating_network.predict_h(test_inputs)
    for k in range(dynamics.mosvgpe.num_experts):
        label = (
            "$\mathbb{E}[h_{"
            + str(dynamics.desired_mode + 1)
            + "}(\mathbf{x}) \mid \mathcal{D}_{0:"
            # + str(iteration)
            + "}]$"
        )
        plot_contf(axs[k, 0], test_inputs, z=mean[:, k])
        # plot_contf(axs[0], test_inputs, z=mean[:, dynamics.desired_mode], label=label)
        label = (
            "$\mathbb{V}[h_{"
            + str(dynamics.desired_mode + 1)
            + "}(\mathbf{x}) \mid \mathcal{D}_{0:"
            # + str(iteration)
            + "}]$"
        )
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


def build_plotting_callbacks(
    dynamics: ModeRLDynamics, logging_epoch_freq: int = 100, num_test: int = 100
) -> List[PlottingCallback]:
    test_inputs = create_test_inputs(num_test=num_test)

    callbacks = [
        PlottingCallback(
            partial(
                plot_gating_network_gps, dynamics=dynamics, test_inputs=test_inputs
            ),
            logging_epoch_freq=logging_epoch_freq,
            name="Gating function posterior",
        ),
        PlottingCallback(
            partial(plot_mixing_probs, dynamics=dynamics, test_inputs=test_inputs),
            logging_epoch_freq=logging_epoch_freq,
            name="Mixing probs",
        ),
    ]
    return callbacks


if __name__ == "__main__":
    save_dirs = {
        "before": "./wandb/run-20220929_123312-2x3tjd8w/files/saved-models/dynamics-before-training-config.json",
        "after": "./wandb/run-20220929_123312-2x3tjd8w/files/saved-models/dynamics-after-training-on-dataset-0-config.json",
    }

    test_inputs = create_test_inputs(100)
    print("test_inputs")
    print(test_inputs.shape)
    print(test_inputs)

    for key in save_dirs.keys():
        dynamics = ModeRLDynamics.load(save_dirs[key])

        plot_gating_networks_gp(dynamics, test_inputs)  # pyright: ignore
        save_name = "./images/gating_network_gp_" + key
        plt.savefig(save_name + ".pdf", transparent=True)
        # tikzplotlib.clean_figure()
        tikzplotlib.save(save_name + ".tex")

        plot_desired_mixing_probs(dynamics, test_inputs)  # pyright: ignore
        save_name = "./images/desired_mixing_prob_" + key
        plt.savefig(save_name + ".pdf", transparent=True)
        # tikzplotlib.clean_figure()
        tikzplotlib.save(save_name + ".tex")
        # plt.show()
        #

        # tikzplotlib.clean_figure()
        # tikzplotlib.save(save_name + ".tex")

    # print("111")
    # print(tikzplotlib.Flavors.latex.preamble())
    # # or
    # print("222")
    # print(tikzplotlib.Flavors.context.preamble())
    # model = keras.models.load_model(
    #     save_dir, custom_objects={"ModeRLDynamics": ModeRLDynamics}
    # )
    # print(type(model))
    # print(model)
    # print(type(model.mosvgpe))
    # print(model.mosvgpe)
    # print(type(model.mosvgpe.experts_list[0]))
    # print(model.mosvgpe.experts_list[0])
    # print(type(model.mosvgpe.gating_network))
    # print(model.mosvgpe.gating_network)
    # gpf.utilities.print_summary(model)
