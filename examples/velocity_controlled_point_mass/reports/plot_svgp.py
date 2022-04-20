#!/usr/bin/env python3
import os

import matplotlib.pyplot as plt
import numpy as np
from modeopt.mode_opt import ModeOpt
from modeopt.plotting import ModeOptContourPlotter
from modeopt.trajectories import GeodesicTrajectory
from velocity_controlled_point_mass.mode_opt_riemannian_energy_traj_opt import (
    create_test_inputs,
)


def generate_trajectories(mode_optimiser):
    dynamics_trajectory = mode_optimiser.dynamics_rollout()[0]
    trajectories = {"dynamics": dynamics_trajectory}
    if mode_optimiser.env is not None:
        trajectories.update({"env": mode_optimiser.env_rollout()})
    if isinstance(
        mode_optimiser.mode_controller.previous_solution,
        GeodesicTrajectory,
    ):
        trajectories.update(
            {"collocation": mode_optimiser.mode_controller.previous_solution.states}
        )
    return trajectories


if __name__ == "__main__":
    experiment_dict = {
        "scenario_7_svgp_baseline_both_modes": "./velocity_controlled_point_mass/experiments/scenario_7/trajectory_optimisation/svgp_baseline_both_modes/2022.04.20/162210/2022-04-20-16-22/ckpts",
        "scenario_7_svgp_baseline_desired_mode": "./velocity_controlled_point_mass/experiments/scenario_7/trajectory_optimisation/svgp_baseline_desired_mode/2022.04.20/162355/2022-04-20-16-24/ckpts",
    }

    save_name = (
        "./velocity_controlled_point_mass/reports/figures/trajectory_optimisation/"
    )
    os.makedirs(save_name, exist_ok=True)

    test_inputs = create_test_inputs(
        x_min=[-3, -3], x_max=[3, 3], input_dim=4, num_test=10000
    )
    for key in experiment_dict.keys():
        mode_optimiser = ModeOpt.load(experiment_dict[key])
        trajectories = generate_trajectories(mode_optimiser)

        mode_optimiser_plotter = ModeOptContourPlotter(
            mode_optimiser=mode_optimiser,
            test_inputs=test_inputs,
            static_trajectories=True,
        )

        figsize = mode_optimiser_plotter.mosvgpe_plotter.figsize
        fig = plt.figure(figsize=(figsize[0] / 2, figsize[1]))
        gs = fig.add_gridspec(1, 1)
        ax = gs.subplots()

        gating_mask = []
        for test_state in mode_optimiser_plotter.test_inputs[:, 0:2]:
            gating_mask.append(
                mode_optimiser.env.state_to_mixing_prob(test_state.numpy())
            )
        gating_mask = np.stack(gating_mask, 0)

        contf = ax.tricontourf(
            mode_optimiser_plotter.test_inputs[:, 0],
            mode_optimiser_plotter.test_inputs[:, 1],
            gating_mask,
            levels=2,
            cmap=mode_optimiser_plotter.mosvgpe_plotter.cmap,
        )
        # mode_optimiser_plotter.plot_env_given_fig(fig)
        mode_optimiser_plotter.plot_trajectories_given_fig(fig)

        plt.savefig(save_name + f"{key}.pdf", transparent=True)
        # plt.show()
