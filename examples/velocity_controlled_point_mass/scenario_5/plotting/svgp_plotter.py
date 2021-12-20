#!/usr/bin/env python3
import numpy as np
import tensorflow as tf
import typing
import palettable
import tensor_annotations.tensorflow as ttf
from attr import attrib
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from tensor_annotations import axes
from modeopt.dynamics import SVGPDynamics

StateControlDim = typing.NewType("StateControlDim", axes.Axis)
NumTest = typing.NewType("NumTest", axes.Axis)

# plt.style.use("science")
# plt.style.use("ggplot")
plt.style.use("seaborn-paper")
# plt.style.use("seaborn")
# plt.style.use("seaborn-dark-palette")

FACTOR = 1.2
NUM_TEST = 10000
CONST_CONTROL = 0


def init_axis_labels_and_ticks(axs):
    if isinstance(axs, np.ndarray):
        for ax in axs.flat:
            ax.set(xlabel="$x$", ylabel="$y$")
        # Hide x labels and tick labels for top plots and y ticks for right plots.
        for ax in axs.flat:
            ax.label_outer()
    else:
        axs.set(xlabel="$x$", ylabel="$y$")
        # Hide x labels and tick labels for top plots and y ticks for right plots.
        axs.label_outer()
    return axs


class SVGPPlotter:
    svgp: SVGPDynamics
    test_inputs: ttf.Tensor2[NumTest, StateControlDim] = attrib()

    @test_inputs.default
    def _default_test_inputs(self):
        sqrtN = int(np.sqrt(NUM_TEST))
        x_min, x_max = (-4.0, 4.0)
        y_min, y_max = (-4.0, 4.0)
        xx = np.linspace(x_min * FACTOR, x_max * FACTOR, sqrtN)
        yy = np.linspace(y_min * FACTOR, y_max * FACTOR, sqrtN)
        xx, yy = np.meshgrid(xx, yy)
        test_inputs = np.column_stack([xx.reshape(-1), yy.reshape(-1)])
        test_inputs = np.concatenate(
            [
                test_inputs,
                np.ones((test_inputs.shape[0], test_inputs.shape[1])) * CONST_CONTROL,
            ],
            -1,
        )
        return test_inputs

    def plot_gp():
        mean_contf, var_contf = self.plot_gp_contf(
            fig, axs, h_means[:, desired_mode], h_vars[:, desired_mode]
        )
        self.add_cbar(
            fig,
            axs[0],
            mean_contf,
            # "Expert "
            # + str(desired_mode + 1)
            # +
            "Gating Function Mean $\mathbb{E}[h_{"
            + str(desired_mode + 1)
            + "}(\mathbf{x})]$",
        )
        self.add_cbar(
            fig,
            axs[1],
            var_contf,
            # "Expert "
            # + str(desired_mode + 1)
            # +
            "Gating Function Variance $\mathbb{V}[h_{"
            + str(desired_mode + 1)
            + "}(\mathbf{x})]$",
        )

    def create_fig_axs_plot_vs_time(self):
        fig = plt.figure(figsize=(self.figsize[0] / 2, self.figsize[1] / 2))
        gs = fig.add_gridspec(1, 1, wspace=0.3)
        ax = gs.subplots(sharex=True, sharey=True)
        ax.set_xlabel("$t$")
        return fig, ax

    def contf(self, fig, ax, z):
        contf = ax.tricontourf(
            self.test_inputs[:, 0],
            self.test_inputs[:, 1],
            z,
            levels=10,
            # levels=var_levels,
            cmap=self.cmap,
        )
        return contf

    def plot_gp_contf(self, fig, axs, mean, var, mean_levels=None, var_levels=None):
        """Plots contours for mean and var side by side"""
        mean_contf = self.contf(fig, axs[0], z=mean)
        var_contf = self.contf(fig, axs[1], z=var)
        return mean_contf, var_contf


def add_cbar(fig, ax, contf, label=""):
    divider = make_axes_locatable(ax)
    # cax = divider.append_axes("right", size="5%", pad=0.05)

    if isinstance(ax, np.ndarray):
        divider = make_axes_locatable(ax[0])
    else:
        divider = make_axes_locatable(ax)
    cax = divider.append_axes("top", size="5%", pad=0.05)
    cbar = fig.colorbar(
        contf,
        # ax=ax,
        use_gridspec=True,
        cax=cax,
        orientation="horizontal",
    )

    cax.xaxis.set_ticks_position("top")
    cax.xaxis.set_label_position("top")
    cbar.set_label(label)
    return cbar


def gating_mask(test_states, env):
    # Calc gating mask at test states
    gating_mask = []
    for test_state in test_states:
        gating_mask.append(env.state_to_mixing_prob(test_state))
    gating_mask = tf.stack(gating_mask, 0)
    return gating_mask


if __name__ == "__main__":
    from simenvs.core import make

    start_state = [3.0, -1.0]
    state_dim = 2
    test_states = mogpe_plotter.test_inputs[:, 0:state_dim]

    # Configure environment
    env_name = "velocity-controlled-point-mass/scenario-5"
    env = make(env_name)
    env.state_init = start_state

    plotter = SVGPPlotter(svgp=svgp)
