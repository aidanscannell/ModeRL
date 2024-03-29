#!/usr/bin/env python3
import logging
import os


logging.basicConfig(level=logging.INFO)

import gpflow as gpf
import hydra
import matplotlib.pyplot as plt
import moderl
import numpy as np
import omegaconf
import tensorflow as tf
import tensorflow_probability as tfp
import wandb
from experiments.plot.callbacks.controller import (
    plot_trajectories_over_desired_gating_gp,
    plot_trajectories_over_desired_mixing_prob,
)
from experiments.plot.utils import create_test_inputs
from experiments.utils import sample_mosvgpe_inducing_inputs_from_data
from gpflow import default_float
from moderl.constraints import build_mode_chance_constraints_scipy
from moderl.controllers import ExplorativeController
from moderl.custom_types import State
from moderl.dynamics import ModeRLDynamics
from moderl.rollouts import collect_data_from_env
from scipy.optimize import LinearConstraint
from wandb.keras import WandbCallback


tfd = tfp.distributions

tf.keras.utils.set_random_seed(42)

logger = logging.getLogger(__name__)
# logger.setLevel(logging.DEBUG)
# logger.setLevel(logging.INFO)


params = {
    # "text.usetex": True,
    # "savefig.transparent": True,
    "image.cmap": "RdYlGn",
    # "figure.figsize": (4, 3),
    # "figure.figsize": (3.5, 2.7),
}
plt.rcParams.update(params)


def set_desired_mode(dynamics: ModeRLDynamics) -> int:
    """Find mode with lowest process noise (uses product of output dims)"""
    noise_var_prod = tf.reduce_prod(
        dynamics.mosvgpe.experts_list[0].gp.likelihood.variance
    )
    desired_mode = 0
    for k, expert in enumerate(dynamics.mosvgpe.experts_list):
        if tf.reduce_prod(expert.gp.likelihood.variance) < noise_var_prod:
            noise_var_prod = expert.gp.likelihood.variance
            desired_mode = k
    logger.info("Desired mode is {}".format(desired_mode))
    return desired_mode


def check_converged(controller: ExplorativeController, target_state: State) -> bool:
    """Returns true if Pr(desired_mode | target_state) > mode_satisfaction_prob"""
    control_zeros = tf.zeros((1, controller.control_dim), dtype=default_float())
    target_input = tf.concat([target_state, control_zeros], -1)
    prob = controller.dynamics.mosvgpe.gating_network.predict_mixing_probs(
        target_input
    )[:, controller.dynamics.desired_mode]
    if prob > controller.mode_satisfaction_prob:
        logger.info("Converged - Agent at target state")
        return True
    else:
        logger.info("Not converged")
        return False


def train(cfg: omegaconf.DictConfig):
    # Make experiment reproducible
    tf.keras.utils.set_random_seed(cfg.training.random_seed)

    # Initialise WandB run
    run = wandb.init(
        entity=cfg.wandb.entity,
        project=cfg.wandb.project,
        config=omegaconf.OmegaConf.to_container(
            cfg, resolve=True, throw_on_missing=True
        ),
        tags=cfg.wandb.tags,
        name=cfg.wandb.run_name,
    )
    log_dir = run.dir
    save_name = os.path.join(
        log_dir, "saved-models/controller-optimised-{}-config.json"
    )

    # Configure environment
    env = hydra.utils.instantiate(cfg.env)

    start_state = hydra.utils.instantiate(cfg.start_state)
    target_state = hydra.utils.instantiate(cfg.target_state)

    # Sample initial data set from env
    initial_dataset = hydra.utils.instantiate(cfg.initial_dataset)

    # Instantiate dynamics model and sample inducing inputs from data
    dynamics = hydra.utils.instantiate(
        cfg.dynamics,
        dataset=initial_dataset,
        callbacks=[
            # DynamicsLoggingCallback(dynamics=dynamics, logging_epoch_freq=5),
            tf.keras.callbacks.EarlyStopping(
                monitor="loss",
                patience=cfg.training.callbacks.patience,
                min_delta=cfg.training.callbacks.min_delta,
                verbose=0,
                restore_best_weights=True,
            ),
            WandbCallback(
                monitor="loss",
                save_graph=False,
                save_model=False,
                save_weights_only=False,
                save_traces=False,
            ),
        ],
    )
    # dynamics.callbacks.append(
    #     build_dynamics_plotting_callbacks(dynamics=dynamics, logging_epoch_freq=5)
    # )
    sample_mosvgpe_inducing_inputs_from_data(
        model=dynamics.mosvgpe, X=initial_dataset[0]
    )
    # Make gating kernel params/undesired modes noise_variance  not trainable
    gpf.utilities.set_trainable(dynamics.mosvgpe.gating_network.gp.kernel, False)

    # Build/train dynamics on initial data set and set desired dynamics mode
    logger.info("Learning dynamics...")
    dynamics.optimise()
    logger.info("Finished learning dynamics with initial dataset")
    dynamics.desired_mode = set_desired_mode(dynamics)

    # Make undesired modes noise_variance not trainable
    # Needed to stop inf in bound due to expert 2 learning very low noise variance
    gpf.utilities.set_trainable(
        dynamics.mosvgpe.experts_list[dynamics.desired_mode].gp.likelihood, False
    )

    # Configure explorative controller (wraps reward_fn in the explorative objective)
    explorative_controller = hydra.utils.instantiate(cfg.controller, dynamics=dynamics)

    explorative_controller.reward_fn = (
        moderl.reward_functions.TargetStateRewardFunction(
            tf.constant([[100.0, 0.0], [0.0, 100.0]], dtype="float64"),
            target_state=target_state,
        )
        + moderl.reward_functions.ControlQuadraticRewardFunction(
            tf.constant([[1.0, 0.0], [0.0, 1.0]], dtype="float64")
        )
        + moderl.reward_functions.StateDiffRewardFunction(
            tf.constant([[1.0, 0.0], [0.0, 1.0]], dtype="float64"),
            target_state=target_state,
        )
    )

    # Run the MBRL loop
    test_inputs = create_test_inputs(40000)  # test inputs for plotting
    explorative_controller.save(save_name.format("before"))
    num_episodes_with_constraint_violations = 0
    initial_delta = 1.0 - explorative_controller.mode_satisfaction_prob
    delta = initial_delta
    initial_exploration_weight = explorative_controller.exploration_weight
    exploration_weight = initial_exploration_weight
    for episode in range(0, cfg.training.num_episodes):
        # Train the dynamics model and set the desired dynamics mode
        if episode > 0:
            logger.info("Learning dynamics...")
            dynamics.optimise()
            logger.info("Finished learning dynamics")

        try:
            # Decay delta (i.e. tighten constraint)
            delta = initial_delta * cfg.constraint_schedule.decay_rate ** (
                episode / cfg.constraint_schedule.decay_episodes
            )
            mode_satisfaction_prob = 1.0 - delta
            constraints = [
                build_mode_chance_constraints_scipy(
                    dynamics=dynamics,
                    control_trajectory=explorative_controller.trajectory_optimiser.previous_solution,
                    start_state=start_state,
                    lower_bound=mode_satisfaction_prob,
                    upper_bound=1.0,  # max prob=1.0
                    # compile=False,
                    compile=True,
                )
            ]
            if (
                explorative_controller.control_lower_bound is not None
                and explorative_controller.control_upper_bound is not None
            ):
                constraints.append(
                    LinearConstraint(
                        np.eye(cfg.controller.horizon * cfg.controller.control_dim),
                        explorative_controller.control_lower_bound,
                        explorative_controller.control_upper_bound,
                    )
                )
            explorative_controller.trajectory_optimiser.constraints = constraints
            logger.info("Updated constraint usinexploration_weightg schedule")
        except:
            logger.info("No constraint schedule")

        try:
            # Decay exploration weight
            exploration_weight = (
                initial_exploration_weight
                * cfg.exploration_weight_schedule.decay_rate
                ** (episode / cfg.exploration_weight_schedule.decay_episodes)
            )
            explorative_controller.trajectory_optimiser.objective_fn = (
                explorative_controller.build_augmentd_objective_fn(exploration_weight)
            )
            logger.info("Updated exploration weight using schedule")
        except:
            logger.info("No exploration weight schedule")

        # Log exploration weight (beta) and constraint (delta)
        wandb.log({"Delta": delta})
        wandb.log({"Beta": exploration_weight})

        # Optimise the constrained objective
        logger.info("Optimising controller...")
        # _ = explorative_controller.optimise()
        opt_result = explorative_controller.optimise()
        logger.info("CONTROLLER OPTIMISATION RESULT")
        logger.info(opt_result)
        logger.info("Finished optimising controller")

        # Rollout the controller in env to collect state transition data
        logger.info("Collecting data from env with controller")
        X, Y = collect_data_from_env(
            env=env, start_state=start_state, controls=explorative_controller()
        )
        dynamics.update_dataset(dataset=(X, Y))

        # Log the extrinsic return ######
        if cfg.wandb.log_extrinsic_return:
            extrinsic_return = explorative_controller.reward_fn(
                state=tfd.Deterministic(X[:, : dynamics.state_dim]),
                control=tfd.Deterministic(X[:, dynamics.state_dim :]),
            )
            # TODO add final state??
            wandb.log({"Extrinsic return": extrinsic_return})

        # Log the number of constraint violations
        num_constraint_violations = 0
        for test_state in X[:, : dynamics.state_dim]:
            pixel = env.state_to_pixel(test_state)
            gating_value = env.gating_bitmap[pixel[0], pixel[1]]
            if gating_value < 0.5:
                num_constraint_violations += 1
        if num_constraint_violations > 0:
            num_episodes_with_constraint_violations += 1
        if cfg.wandb.log_constraint_violations:
            wandb.log({"Num constraint violations": num_constraint_violations})
            wandb.log(
                {
                    "Num episodes with constraint violations": num_episodes_with_constraint_violations
                }
            )

        # Plot trajectory over learned dynamics
        if cfg.wandb.log_artifacts:
            fig = plot_trajectories_over_desired_mixing_prob(
                env,
                controller=explorative_controller,
                test_inputs=test_inputs,
                target_state=target_state,
            )
            wandb.log({"Final traj over desired mixing prob": wandb.Image(fig)})
            fig = plot_trajectories_over_desired_gating_gp(
                env,
                controller=explorative_controller,
                test_inputs=test_inputs,
                target_state=target_state,
            )
            wandb.log({"Final traj over desired gating gp": wandb.Image(fig)})

        # Save the controller
        if cfg.training.save:
            explorative_controller.save(save_name.format(episode))

        # if check_converged(explorative_controller, target_state=target_state):
        #     # TODO implement a better check for convergence
        #     break
        distance_from_target_state = np.linalg.norm(
            (X[-1, 0 : dynamics.state_dim] - target_state), axis=-1
        )
        if distance_from_target_state < 0.05 and num_constraint_violations == 0:
            logger.info(
                "Termination criteria met (<0.05), ||x - target_state||^2)={}".format(
                    distance_from_target_state
                )
            )
            break
        else:
            logger.info(
                "Termination criteria NOT met (<0.05), ||x - target_state||^2)={}".format(
                    distance_from_target_state
                )
            )


if __name__ == "__main__":
    train()  # pyright: ignore
