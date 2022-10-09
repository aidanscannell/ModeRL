#!/usr/bin/env python3
from collections import OrderedDict
from typing import Callable, Optional

import matplotlib
import matplotlib.pyplot as plt
import palettable
import tensorflow as tf
import wandb
from experiments.plot.utils import create_test_inputs, plot_contf
from moderl.controllers import ControllerInterface
from moderl.custom_types import InputData, State
from moderl.rollouts import rollout_trajectory_optimisation_controller_in_env


CMAP = palettable.scientific.sequential.Bilbao_15.mpl_colormap

LABELS = {"env": "Environment", "dynamics": "Dynamics"}
COLORS = {"env": "c", "dynamics": "m"}
LINESTYLES = {"env": "-", "dynamics": "-"}
MARKERS = {"env": "*", "dynamics": "."}


PlotFn = Callable[[], matplotlib.figure.Figure]


class WandBImageCallbackScipy:
    def __init__(
        self, plot_fn: PlotFn, logging_epoch_freq: int = 10, name: Optional[str] = ""
    ):
        self.plot_fn = plot_fn
        self.logging_epoch_freq = logging_epoch_freq
        self.name = name

    def __call__(self, step, variables, value):
        if step % self.logging_epoch_freq == 0:
            fig = self.plot_fn()
            wandb.log({self.name: wandb.Image(fig)})


def plot_trajectories_over_desired_mixing_prob(
    env, controller: ControllerInterface, test_inputs: InputData, target_state: State
):
    fig = plt.figure()
    gs = fig.add_gridspec(1, 1)
    ax = gs.subplots()
    probs = controller.dynamics.mosvgpe.gating_network.predict_mixing_probs(test_inputs)
    plot_contf(ax, test_inputs, z=probs[:, controller.dynamics.desired_mode])
    plot_trajectories(ax, env, controller=controller, target_state=target_state)
    plot_env(ax, env=env, test_inputs=test_inputs)
    plot_mode_satisfaction_probability_given_ax(
        ax, controller=controller, test_inputs=test_inputs
    )
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    fig.tight_layout()
    fig.legend(
        by_label.values(),
        by_label.keys(),
        bbox_to_anchor=(0.5, 0.05),
        loc="upper center",
        bbox_transform=fig.transFigure,
        ncol=len(by_label),
    )
    return fig


def plot_trajectories_over_desired_gating_gp(
    env, controller: ControllerInterface, test_inputs: InputData, target_state: State
):
    fig = plt.figure()
    gs = fig.add_gridspec(1, 2)
    axs = gs.subplots()
    mean, var = controller.dynamics.mosvgpe.gating_network.predict_h(test_inputs)
    plot_contf(axs[0], test_inputs, z=mean[:, controller.dynamics.desired_mode])
    plot_contf(axs[1], test_inputs, z=var[:, controller.dynamics.desired_mode])

    for ax in axs:
        plot_trajectories(ax, env=env, controller=controller, target_state=target_state)
        plot_env(ax, env=env, test_inputs=test_inputs)
        plot_mode_satisfaction_probability_given_ax(
            ax, controller=controller, test_inputs=test_inputs
        )
    handles, labels = fig.gca().get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    fig.tight_layout()
    fig.legend(
        by_label.values(),
        by_label.keys(),
        bbox_to_anchor=(0.5, 0.05),
        loc="upper center",
        bbox_transform=fig.transFigure,
        ncol=len(by_label),
    )
    return fig


def plot_trajectories(ax, env, controller: ControllerInterface, target_state: State):
    env_traj = rollout_trajectory_optimisation_controller_in_env(
        env=env, start_state=controller.start_state, controller=controller
    )
    dynamics_traj = controller.rollout_in_dynamics().mean()

    for traj, key in zip([env_traj, dynamics_traj], ["env", "dynamics"]):
        # for traj, key in zip([dynamics_traj], ["dynamics"]):
        ax.plot(
            traj[:, 0],
            traj[:, 1],
            label=LABELS[key],
            color=COLORS[key],
            linestyle=LINESTYLES[key],
            linewidth=0.3,
            marker=MARKERS[key],
        )
    plot_start_end_pos(
        ax, start_state=controller.start_state, target_state=target_state
    )


def plot_mode_satisfaction_probability_given_ax(
    ax, controller: ControllerInterface, test_inputs: InputData
):
    mixing_probs = controller.dynamics.mosvgpe.gating_network.predict_mixing_probs(
        test_inputs
    )
    CS = ax.tricontour(
        test_inputs[:, 0],
        test_inputs[:, 1],
        mixing_probs[:, controller.dynamics.desired_mode].numpy(),
        [controller.mode_satisfaction_prob],
    )
    ax.clabel(CS, inline=True, fontsize=10)


def plot_env(ax, env, test_inputs: InputData):
    test_states = test_inputs[:, 0:2]
    mode_probs = []
    for test_state in test_states:
        pixel = env.state_to_pixel(test_state)
        mode_probs.append(env.gating_bitmap[pixel[0], pixel[1]])
    mode_probs = tf.stack(mode_probs, 0)
    CS = ax.tricontour(
        test_states[:, 0], test_states[:, 1], mode_probs.numpy(), [0.5], colors=["k"]
    )
    try:
        # clabel = ax.clabel(CS, inline=True, fmt={0.5: "Mode boundary"})
        clabel = ax.clabel(CS, inline=True, fontsize=8, fmt={0.5: "Mode boundary"})
        clabel[0].set_bbox(dict(boxstyle="round,pad=0.1", fc="white", alpha=1.0))
    except IndexError:
        pass


def plot_env_cmap(ax, env, test_inputs: InputData, cmap=CMAP):
    test_states = test_inputs[:, 0:2]
    mode_probs = []
    for test_state in test_states:
        pixel = env.state_to_pixel(test_state)
        mode_probs.append(env.gating_bitmap[pixel[0], pixel[1]])
    mode_probs = tf.stack(mode_probs, 0)
    print("mode_probs")
    print(mode_probs)
    print(tf.reduce_min(mode_probs))
    print(tf.reduce_max(mode_probs))
    ax.tricontour(
        test_states[:, 0],
        test_states[:, 1],
        mode_probs.numpy(),
        # [-1, 2],
        # [0.1, 0.9],
        cmap=cmap,
    )


def plot_start_end_pos(ax, start_state, target_state):
    # def plot_start_end_pos(ax, start_state, target_state, bbox=False):
    # if bbox:
    #     bbox = dict(boxstyle="round,pad=0.1", fc="thistle", alpha=1.0)
    # else:
    #     bbox = None
    bbox = dict(boxstyle="round,pad=0.1", fc="thistle", alpha=1.0)
    if len(start_state.shape) == 1:
        start_state = start_state[tf.newaxis, :]
    if len(target_state.shape) == 1:
        target_state = target_state[tf.newaxis, :]
    ax.annotate(
        r"$\mathbf{s}_0$",
        (start_state[0, 0] + 0.15, start_state[0, 1]),
        horizontalalignment="left",
        verticalalignment="top",
        bbox=bbox,
    )
    ax.annotate(
        r"$\mathbf{s}_f$",
        (target_state[0, 0] + 0.15, target_state[0, 1]),
        horizontalalignment="left",
        verticalalignment="bottom",
        bbox=bbox,
    )
    ax.scatter(start_state[0, 0], start_state[0, 1], marker="x", color="k", s=8.0)
    ax.scatter(
        target_state[0, 0],
        target_state[0, 1],
        color="k",
        marker="x",
        s=8.0,
    )


def build_controller_plotting_callback(
    env,
    controller: ControllerInterface,
    target_state: State,
    logging_epoch_freq: int = 10,
    num_test: int = 100,
) -> Callable:
    test_inputs = create_test_inputs(num_test=num_test)

    def plotting_callback(step, variables, value):
        if step % logging_epoch_freq == 0:
            fig = plot_trajectories_over_desired_gating_gp(
                env=env,
                controller=controller,
                test_inputs=test_inputs,
                target_state=target_state,
            )
            wandb.log({"Trajectories over desired gating GP": wandb.Image(fig)})
            fig = plot_trajectories_over_desired_mixing_prob(
                env=env,
                controller=controller,
                test_inputs=test_inputs,
                target_state=target_state,
            )
            wandb.log({"Trajectories over desired mixing prob": wandb.Image(fig)})

    return plotting_callback


# if __name__ == "__main__":
#     save_dirs = {
#         "before": "./wandb/run-20220929_123312-2x3tjd8w/files/saved-models/dynamics-before-training-config.json",  # noqa: E501
#         "after": "./wandb/run-20220929_123312-2x3tjd8w/files/saved-models/dynamics-after-training-on-dataset-0-config.json",  # noqa: E501
#     }

#     test_inputs = create_test_inputs(100)
#     print("test_inputs")
#     print(test_inputs.shape)
#     print(test_inputs)

#     for key in save_dirs.keys():
#         dynamics = ModeRLDynamics.load(save_dirs[key])
#         explorative_controller = ExplorativeController.load(save_dirs[key])

#         plot_gating_networks_gp(dynamics, test_inputs)  # pyright: ignore
#         save_name = "./images/gating_network_gp_" + key
#         plt.savefig(save_name + ".pdf", transparent=True)
#         # tikzplotlib.clean_figure()

#         plot_desired_mixing_probs(dynamics, test_inputs)  # pyright: ignore
#         save_name = "./images/desired_mixing_prob_" + key
#         plt.savefig(save_name + ".pdf", transparent=True)
#         # tikzplotlib.clean_figure()
#         )
