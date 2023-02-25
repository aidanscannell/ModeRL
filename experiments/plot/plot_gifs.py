#!/usr/bin/env python3
import argparse
import pathlib
from functools import partial

import hydra
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import wandb
from experiments.plot.utils import (
    create_test_inputs,
    get_ExplorativeController_from_id,
    plot_contf,
    plot_env,
    plot_mode_satisfaction_prob,
    plot_trajectories,
)
from figures import custom_labels
from matplotlib.animation import FuncAnimation
from moderl.custom_types import State
from omegaconf import OmegaConf


tfd = tfp.distributions
params = {
    "text.usetex": True,
    "savefig.transparent": True,
    "savefig.bbox": "tight",
    "image.cmap": "bwr_r",
    "figure.figsize": (3.5, 2.5),
    "text.latex.preamble": r"\usepackage{amsfonts}",
}
plt.rcParams.update(params)


def plot_frame(
    i: int,
    ax,
    environment,
    run_id: str,
    wandb_dir: str,
    target_state: State,
    test_inputs,
):
    print("Plotting iteration: {}".format(i))
    ax.clear()
    ax.set_xlim(np.min(test_inputs[:, 0]), np.max(test_inputs[:, 0]))
    ax.set_ylim(np.min(test_inputs[:, 1]), np.max(test_inputs[:, 1]))

    levels = np.linspace(0, 1, 11)

    plot_env(ax, environment, test_inputs=test_inputs)
    explorative_controller = get_ExplorativeController_from_id(
        i, id=run_id, wandb_dir=wandb_dir
    )
    probs = explorative_controller.dynamics.mosvgpe.gating_network.predict_mixing_probs(
        test_inputs
    )
    contf = plot_contf(
        ax,
        test_inputs,
        z=probs[:, explorative_controller.dynamics.desired_mode],
        levels=levels,
    )
    plot_mode_satisfaction_prob(
        ax, controller=explorative_controller, test_inputs=test_inputs
    )
    plot_trajectories(
        ax,
        environment,
        controller=explorative_controller,
        target_state=target_state,
    )

    ax.set_xticklabels([])
    ax.set_yticklabels([])

    # divider = make_axes_locatable(ax)
    # cax = divider.append_axes("right", size="5%", pad=0.05)
    # cbar = fig.colorbar(contf, use_gridspec=True, cax=cax)
    # cbar.set_label(
    #     r"$\Pr(\alpha=k^* \mid \mathbf{s}, \mathcal{D}_{0:" + str(iteration) + "})$"
    # )

    by_label = custom_labels(ax)
    ax.legend(by_label.values(), by_label.keys(), loc="lower left")
    # fig.tight_layout()


def plot_gif(
    run_id,
    save_name: str,
    wandb_dir: str,
    num_episodes: int = 60,
    random_seed: int = 60,
):
    api = wandb.Api()
    run = api.run(run_id)
    tf.keras.utils.set_random_seed(random_seed)
    run_id = run_id.split("/")[-1]

    cfg = OmegaConf.create(run.config)
    env = hydra.utils.instantiate(cfg.env)
    target_state = hydra.utils.instantiate(cfg.target_state)

    # fig, ax = plt.subplots(figsize=(6, 6))
    fig, ax = plt.subplots()
    update_custom = partial(
        plot_frame,
        ax=ax,
        environment=env,
        run_id=run_id,
        wandb_dir=wandb_dir,
        target_state=target_state,
        test_inputs=test_inputs,
    )

    anime = FuncAnimation(
        fig=fig,
        func=update_custom,
        frames=num_episodes,
        interval=250
        # interval=150
    )
    pathlib.Path(save_name).parents[0].mkdir(parents=True, exist_ok=True)
    anime.save(save_name)


if __name__ == "__main__":
    test_inputs = create_test_inputs(num_test=40000)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--wandb_dir",
        help="directory contraining wandb results",
        default="./wandb",
    )
    parser.add_argument(
        "--saved_runs_yaml",
        help="yaml file containing saved runs",
        default="./saved_runs.yaml",
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        help="random seed to fix stochasticity of environment",
        default=42,
    )
    args = parser.parse_args()

    saved_runs = OmegaConf.load(args.saved_runs_yaml)
    run_ids = {
        "moderl-exploration.gif": {"id": saved_runs.moderl.id, "num_episodes": 60},
        "moderl-constraint-schedule-exploration.gif": {
            "id": saved_runs.moderl_schedule.id,
            "num_episodes": 60,
        },
        "greedy-with-constraint.gif": {
            "id": saved_runs.greedy_with_constraint.id,
            "num_episodes": 20,
        },
        "greedy-no-constraint.gif": {
            "id": saved_runs.greedy_no_constraint.id,
            "num_episodes": 20,
        },
        "aleatoric-uncertainty.gif": {
            "id": saved_runs.aleatoric_unc_ablation.id,
            "num_episodes": 20,
        },
        "myopic-moderl.gif": {"id": saved_runs.myopic_ablation.id, "num_episodes": 40},
    }
    # for key in run_ids.keys():
    #     plot_gif(
    #         run_id=run_ids[key],
    #         save_name="gifs/" + key,
    #         wandb_dir=args.wandb_dir,
    #         num_episodes=1,
    #         random_seed=args.random_seed,
    #     )

    for key in run_ids.keys():
        plot_gif(
            run_id=run_ids[key]["id"],
            save_name="gifs/" + key,
            wandb_dir=args.wandb_dir,
            num_episodes=run_ids[key]["num_episodes"],
            random_seed=args.random_seed,
        )
