#!/usr/bin/env python3
import os
import logging

import gpflow as gpf
import hydra
import omegaconf
import tensorflow as tf
import tensorflow_probability as tfp
from moderl.dynamics import ModeRLDynamics
from moderl.controllers import ExplorativeController
from moderl.objectives import (
    bald_objective,
    conditional_gating_function_entropy,
    independent_gating_function_entropy,
    joint_gating_function_entropy,
)
from moderl.rollouts import collect_data_from_env
from wandb.keras import WandbCallback

import wandb
from experiments.callbacks import DynamicsLoggingCallback
from experiments.plot.controller import (
    plot_trajectories_over_desired_gating_gp,
    plot_trajectories_over_desired_mixing_prob,
)
from experiments.plot.utils import create_test_inputs
from experiments.utils import sample_mosvgpe_inducing_inputs_from_data

tfd = tfp.distributions

tf.keras.utils.set_random_seed(42)

EXPLORATIVE_OBJECTIVE_FNS = {
    "joint_gating_function_entropy": joint_gating_function_entropy,
    "bald": bald_objective,
    "independent_gating_function_entropy": independent_gating_function_entropy,
    "greedy": lambda *args, **kwargs: 0.0,  # i.e. no exploration term
    "conditional_gating_function_entropy": conditional_gating_function_entropy,
}


def set_desired_mode(dynamics: ModeRLDynamics) -> int:
    """Sets desired mode to one with lowest process noise (uses product of output dims)"""
    noise_var_prod = tf.reduce_prod(
        dynamics.mosvgpe.experts_list[0].gp.likelihood.variance
    )
    desired_mode = 0
    for k, expert in enumerate(dynamics.mosvgpe.experts_list):
        if tf.reduce_prod(expert.gp.likelihood.variance) < noise_var_prod:
            noise_var_prod = expert.gp.likelihood.variance
            desired_mode = k
    print("Desired mode is {}".format(desired_mode))
    return desired_mode


@hydra.main(config_path="configs", config_name="main")
def run_experiment(cfg: omegaconf.DictConfig):
    ###### Make experiment reproducible ######
    tf.keras.utils.set_random_seed(cfg.experiment.random_seed)

    ###### Initialise WandB run ######
    run = wandb.init(
        entity=cfg.wandb.entity,
        project=cfg.wandb.project,
        config=omegaconf.OmegaConf.to_container(
            cfg, resolve=True, throw_on_missing=True
        ),
    )
    log_dir = run.dir
    save_name = os.path.join(
        log_dir, "saved-models/controller-optimised-{}-config.json"
    )

    logging.info(
        "Using {} for explorative objective".format(
            cfg.experiment.explorative_objective_fn
        )
    )

    ###### Configure environment ######
    env = hydra.utils.instantiate(cfg.env)

    start_state = hydra.utils.instantiate(cfg.start_state)
    target_state = hydra.utils.instantiate(cfg.target_state)

    ###### Sample initial data set from env ######
    initial_dataset = hydra.utils.instantiate(cfg.initial_dataset)

    ###### Instantiate dynamics model and sample inducing inputs from data ######
    dynamics = hydra.utils.instantiate(
        cfg.dynamics,
        dataset=initial_dataset,
        callbacks=[
            # dynamics.callbacks = [
            # DynamicsLoggingCallback(dynamics=dynamics, logging_epoch_freq=5),
            # build_plotting_callbacks(
            #     dynamics=dynamics, logging_epoch_freq=logging_epoch_freq
            # ),
            tf.keras.callbacks.EarlyStopping(
                monitor="loss",
                patience=cfg.experiment.callbacks.patience,
                min_delta=cfg.experiment.callbacks.min_delta,
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
    sample_mosvgpe_inducing_inputs_from_data(
        model=dynamics.mosvgpe, X=initial_dataset[0]
    )

    ###### Build/train dynamics on initial data set and set desired dynamics mode ######
    dynamics.optimise()
    dynamics.desired_mode = set_desired_mode(dynamics)

    ###### Make gating kernel params/undesired modes noise_variance  not trainable ######
    gpf.utilities.set_trainable(dynamics.mosvgpe.gating_network.gp.kernel, False)
    gpf.utilities.set_trainable(
        dynamics.mosvgpe.experts_list[dynamics.desired_mode].gp.likelihood, False
    )  # Needed to stop inf in bound due to expert 2 learning very low noise variance

    ###### Configure the explorative controller (wraps reward_fn in the explorative objective) ######
    explorative_objective_fn = EXPLORATIVE_OBJECTIVE_FNS[
        cfg.experiment.explorative_objective_fn
    ]
    explorative_controller = hydra.utils.instantiate(
        cfg.controller,
        dynamics=dynamics,
        explorative_objective_fn=explorative_objective_fn,
    )

    ###### Run the mbrl loop ######
    converged = False
    test_inputs = create_test_inputs(40000)  # test inputs for plotting
    explorative_controller.save(save_name.format("before"))
    for episode in range(0, cfg.experiment.num_episodes):
        ###### Train the dynamics model and set the desired dynamics mode ######
        if episode > 0:
            dynamics.optimise()

        ###### Optimise the constrained objective ######
        _ = explorative_controller.optimise()

        if converged:
            # TODO implement check for convergence
            break

        ###### Rollout the controller in env to collect state transition data ######
        X, Y = collect_data_from_env(
            env=env,
            start_state=start_state,
            controls=explorative_controller(),
        )
        dynamics.update_dataset(dataset=(X, Y))

        ###### Log the extrinsic return ######
        if cfg.wandb.log_extrinsic_return:
            extrinsic_return = explorative_controller.reward_fn(
                state=tfd.Deterministic(X[:, : dynamics.state_dim]),
                control=tfd.Deterministic(X[:, dynamics.state_dim :]),
            )
            # TODO add final state??
            wandb.log({"Extrinsic return": extrinsic_return})

        ###### Log the number of constraint violations ######
        if cfg.wandb.log_constraint_violations:
            num_constraint_violations = 0.0
            for test_state in X[:, : dynamics.state_dim]:
                pixel = env.state_to_pixel(test_state)
                gating_value = env.gating_bitmap[pixel[0], pixel[1]]
                if gating_value < 0.5:
                    num_constraint_violations += 1.0
            wandb.log({"Number constraint violations": num_constraint_violations})

        ###### Plot trajectory over learned dynamics ######
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

        ###### Save the controller ######
        if cfg.experiment.save:
            explorative_controller.save(save_name.format(episode))


if __name__ == "__main__":
    run_experiment()  # pyright: ignore
