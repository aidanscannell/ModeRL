#!/usr/bin/env python3
import argparse

import hydra
import matplotlib.pyplot as plt
import tensorflow as tf
import wandb
from experiments.plot.figures import plot_four_iterations_in_row, plot_sinlge_run
from omegaconf import OmegaConf


params = {
    "text.usetex": True,
    "savefig.transparent": True,
    "image.cmap": "RdYlGn",
    "figure.figsize": (4, 3),
    # "figure.figsize": (3.5, 2.7),
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

    for key in saved_runs_dict.keys():
        print("Plotting {}".format(key))
        run_id = saved_runs_dict[key]["id"].split("/")[-1]
        fig = plot_sinlge_run(
            env=env,
            run_id=run_id,
            wandb_dir=wandb_dir,
            target_state=target_state,
            iteration=saved_runs_dict[key]["iteration"],
            # title=saved_runs_dict[key]["title"],
        )
        plt.savefig(os.path.join("./figures", key + ".pdf"))

    fig = plot_four_iterations_in_row(
        env=env,
        run_id=saved_runs.joint_gating.id.split("/")[-1],
        wandb_dir=wandb_dir,
        target_state=target_state,
        iterations=saved_runs.joint_gating.iterations,
        # title=""
    )
    plt.savefig("./figures/joint_gating_four_iterations_in_row.pdf")


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
