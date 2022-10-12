#!/usr/bin/env python3
from typing import List

import matplotlib.pyplot as plt
from experiments.plot.utils import (
    create_test_inputs,
    get_ExplorativeController_from_id,
    plot_env,
    plot_start_end_pos,
)
from moderl.custom_types import State


def figure_1(
    env,
    run_id: str,
    wandb_dir: str,
    target_state: State,
    iterations: List[int] = [0, 1, 2],
):
    test_inputs = create_test_inputs(num_test=40000)
    test_states = test_inputs[:, 0:2]
    fig = plt.figure()
    gs = fig.add_gridspec(1, 1)
    ax = gs.subplots()
    plot_env(ax, env, test_inputs=test_inputs)
    plot_env_cmap(ax, env, test_inputs=test_inputs, cmap=CMAP)

    # contf = plot_contf(
    #     ax, test_inputs, z=probs[:, dynamics.desired_mode], levels=levels
    # )

    for i in iterations:
        explorative_controller = get_ExplorativeController_from_id(
            i=i, id=run_id, wandb_dir=wandb_dir
        )

        probs = (
            explorative_controller.dynamics.mosvgpe.gating_network.predict_mixing_probs(
                test_inputs
            )[:, explorative_controller.dynamics.desired_mode]
        )
        CS = ax.tricontour(
            test_states[:, 0],
            test_states[:, 1],
            probs.numpy(),
            [explorative_controller.mode_satisfaction_prob],
        )
        clabel = ax.clabel(
            CS,
            inline=True,
            fontsize=8,
            fmt={explorative_controller.mode_satisfaction_prob: "i=" + str(i)},
        )
        clabel[0].set_bbox(dict(boxstyle="round,pad=0.1", fc="white", alpha=1.0))
        # clabel = ax.clabel(CS, fmt={0.5: "i=" + str(i)})
        # clabel[0].set_bbox(dict(boxstyle="round,pad=0.1", fc="white", alpha=1.0))

    plot_start_end_pos(
        ax, start_state=explorative_controller.start_state, target_state=target_state
    )
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    fig.tight_layout()
    return fig
