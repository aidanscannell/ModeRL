#!/usr/bin/env python3
import argparse
import os

import hydra
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import wandb
from experiments.plot.figures import (
    plot_constraint_expanding_figure,
    plot_four_iterations_in_row,
    plot_greedy_and_myopic_comparison_figure,
    plot_sinlge_run_gating_function_variance,
    plot_sinlge_run_mode_prob,
    plot_uncertainty_comparison,
)
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from omegaconf import OmegaConf


params = {
    "text.usetex": True,
    "savefig.transparent": True,
    "savefig.bbox": "tight",
    "image.cmap": "PiYG",
    # "image.cmap": "RdYlGn",
    "figure.figsize": (3.5, 2.5),
    # "figure.figsize": (4, 3),
    # "figure.figsize": (3.5, 2.7),
    "text.latex.preamble": r"\usepackage{amsfonts}",
}
plt.rcParams.update(params)


def plot_all_figures(wandb_dir, saved_runs_yaml, random_seed: int = 42):
    tf.keras.utils.set_random_seed(random_seed)
    api = wandb.Api()
    saved_runs = OmegaConf.load(saved_runs_yaml)
    saved_runs_dict = OmegaConf.to_object(saved_runs)

    run = api.run(saved_runs.joint_gating.id)
    cfg = OmegaConf.create(run.config)
    env = hydra.utils.instantiate(cfg.env)
    target_state = hydra.utils.instantiate(cfg.target_state)

    cmap = ListedColormap(["darkred", "lightgrey"])
    # cmap = ListedColormap(["red", "green"])
    viridis = cmap.resampled(256)
    # viridis = mpl.colormaps['viridis'].resampled(256)
    newcolors = viridis(np.linspace(0, 1, 256))
    # pink = np.array([248/256, 24/256, 148/256, 1])
    # newcolors[:25, :] = pink
    newcmap = ListedColormap(newcolors, name="RedGreen")
    mpl.colormaps.register(newcmap)

    top = mpl.colormaps["Reds_r"].resampled(128)
    bottom = mpl.colormaps["Blues"].resampled(128)
    newcolors = np.vstack((top(np.linspace(0, 1, 128)), bottom(np.linspace(0, 1, 128))))
    newcmap = ListedColormap(newcolors, name="OrangeBlue")

    mpl.colormaps.register(newcmap)
    # mpl.colormaps.register(LinearSegmentedColormap('BlueRed2', cdict2))
    # blue_red1 = LinearSegmentedColormap("BlueRed1", cdict1)
    # params = {"image.cmap": "PiYG"}
    params = {"image.cmap": "OrangeBlue"}
    params = {"image.cmap": "bwr_r"}
    # params = {"image.cmap": "RedGreen"}
    plt.rcParams.update(params)
    # cmap=palettable.scientific.sequential.Bilbao_15.mpl_colormap
    # cmap="Reds"
    # cmap = plt.cm.get_cmap("Reds")
    # cmap = plt.cm.get_cmap("viridis")
    # cmap = mpl.colormaps["viridis"]
    # cmap = mpl.colormaps["Reds"]
    # cmap = mpl.cm.get_cmap("viridis")
    # Figure 1
    fig = plot_constraint_expanding_figure(
        env=env,
        run_id=saved_runs.joint_gating.id.split("/")[-1],
        wandb_dir=wandb_dir,
        target_state=target_state,
        iterations=saved_runs.joint_gating.iterations,
    )
    plt.savefig("./figures/joint_gating_constraint_expanding.pdf")
    # exit()

    # Figure 2
    fig = plot_four_iterations_in_row(
        env=env,
        run_id=saved_runs.joint_gating.id.split("/")[-1],
        wandb_dir=wandb_dir,
        target_state=target_state,
        iterations=saved_runs.joint_gating.iterations,
        # title=""
    )
    plt.savefig("./figures/joint_gating_four_iterations_in_row.pdf")

    # Figure 3
    fig = plot_greedy_and_myopic_comparison_figure(
        env,
        saved_runs=saved_runs,
        wandb_dir=wandb_dir,
        target_state=target_state,
        # iteration: int = 0,
    )
    plt.savefig("./figures/greedy_and_myopic_comparisons.pdf")

    # Figure 4
    fig = plot_uncertainty_comparison(
        env=env,
        saved_runs=saved_runs,
        wandb_dir=wandb_dir,
        target_state=target_state,
        # iterations=saved_runs.joint_gating.iterations,
    )
    plt.savefig("./figures/uncertainty_comparison.pdf")

    # for key in saved_runs_dict.keys():
    #     print("Plotting {}".format(key))
    #     run_id = saved_runs_dict[key]["id"].split("/")[-1]
    #     if key is "greedy-with-constraint":
    #         cbar = True
    #         legend = False
    #     else:
    #         cbar = False
    #         legend = True
    #     fig = plot_sinlge_run_mode_prob(
    #         env=env,
    #         run_id=run_id,
    #         wandb_dir=wandb_dir,
    #         target_state=target_state,
    #         iteration=saved_runs_dict[key]["iteration"],
    #         cbar=cbar,
    #         legend=legend,
    #     )
    #     plt.savefig(os.path.join("./figures", key + ".pdf"))
    #     fig = plot_sinlge_run_gating_function_variance(
    #         env=env,
    #         run_id=run_id,
    #         wandb_dir=wandb_dir,
    #         target_state=target_state,
    #         iteration=saved_runs_dict[key]["iteration"],
    #     )
    #     plt.savefig(os.path.join("./figures", key + "_gating_variance.pdf"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--wandb_dir", help="directory contraining wandb results", default="./triton"
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

    plot_all_figures(
        wandb_dir=args.wandb_dir,
        saved_runs_yaml=args.saved_runs_yaml,
        random_seed=args.random_seed,
    )
