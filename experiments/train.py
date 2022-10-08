#!/usr/bin/env python3
import os
from typing import Callable, List, Optional

import gpflow as gpf
import hydra

# import matplotlib.pyplot as plt
import numpy as np
import omegaconf
import simenvs

# import tensor_annotations.tensorflow as ttf
import tensorflow as tf
import tensorflow_probability as tfp
from gpflow import default_float, inducing_variables
from moderl.controllers import ControllerInterface, ExplorativeController
from moderl.custom_types import Batch, InputData, One, StateDim
from moderl.dynamics import ModeRLDynamics
from moderl.objectives import (
    bald_objective,
    independent_gating_function_entropy,
    joint_gating_function_entropy,
)
from moderl.rollouts import collect_data_from_env
from tf_agents.environments import py_environment
from wandb.keras import WandbCallback

import wandb
from experiments.plot.controller import (
    build_controller_plotting_callback,
    plot_trajectories_over_desired_gating_gp,
    plot_trajectories_over_desired_mixing_prob,
)
from experiments.plot.dynamics import build_plotting_callbacks
from experiments.plot.utils import create_test_inputs
from experiments.utils import (
    model_from_DictConfig,
    sample_env_trajectories,
    sample_inducing_inputs_from_data,
    sample_mosvgpe_inducing_inputs_from_data,
)

tfd = tfp.distributions

tf.keras.utils.set_random_seed(42)
# from moderl.mode_rl import mode_rl_loop

# TODO loading model from checkpoint


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


def dynamics_callbacks(
    dynamics: ModeRLDynamics, logging_epoch_freq: int
) -> List[tf.keras.callbacks.Callback]:
    """Configure the dynamics model WandbCallback for experiment tracking/plotting"""
    return [
        build_plotting_callbacks(
            dynamics=dynamics, logging_epoch_freq=logging_epoch_freq
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="loss",
            min_delta=0,
            patience=1200,
            verbose=0,
            mode="auto",
            baseline=None,
            restore_best_weights=False,
        ),
        WandbCallback(
            monitor="val_loss",
            save_graph=False,
            save_model=False,
            save_weights_only=False,
            save_traces=False,
            # save_weights_only=True,
            # log_weights=False,
            # log_evaluation=True,
            # validation_steps=5,
        ),
        DynamicsLoggingCallback(dynamics=dynamics),
    ]


class DynamicsLoggingCallback(tf.keras.callbacks.Callback):
    def __init__(
        self, dynamics: ModeRLDynamics, logging_epoch_freq: int = 10, name: str = ""
    ):
        self.dynamics = dynamics
        self.logging_epoch_freq = logging_epoch_freq
        self.name = name

    def on_epoch_end(self, epoch: int, logs=None):
        if epoch % self.logging_epoch_freq == 0:
            wandb.log(
                {
                    "expert_1_noise_var_1": self.dynamics.mosvgpe.experts_list[
                        0
                    ].gp.likelihood.variance[0]
                }
            )
            wandb.log(
                {
                    "expert_1_noise_var_2": self.dynamics.mosvgpe.experts_list[
                        0
                    ].gp.likelihood.variance[1]
                }
            )
            wandb.log(
                {
                    "expert_2_noise_var_1": self.dynamics.mosvgpe.experts_list[
                        1
                    ].gp.likelihood.variance[0]
                }
            )
            wandb.log(
                {
                    "expert_2_noise_var_2": self.dynamics.mosvgpe.experts_list[
                        1
                    ].gp.likelihood.variance[1]
                }
            )
            wandb.log(
                {
                    "expert_1_kernel_var_1": self.dynamics.mosvgpe.experts_list[0]
                    .gp.kernel.kernels[0]
                    .variance.numpy()
                }
            )
            wandb.log(
                {
                    "expert_1_kernel_var_2": self.dynamics.mosvgpe.experts_list[0]
                    .gp.kernel.kernels[1]
                    .variance.numpy()
                }
            )
            wandb.log(
                {
                    "expert_2_kernel_var_1": self.dynamics.mosvgpe.experts_list[1]
                    .gp.kernel.kernels[0]
                    .variance.numpy()
                }
            )
            wandb.log(
                {
                    "expert_2_kernel_var_2": self.dynamics.mosvgpe.experts_list[1]
                    .gp.kernel.kernels[1]
                    .variance.numpy()
                }
            )
            wandb.log(
                {
                    "expert_1_kernel_lengthscale_1": self.dynamics.mosvgpe.experts_list[
                        0
                    ]
                    .gp.kernel.kernels[0]
                    .lengthscales.numpy()[0]
                }
            )
            wandb.log(
                {
                    "expert_1_kernel_lengthscale_2": self.dynamics.mosvgpe.experts_list[
                        0
                    ]
                    .gp.kernel.kernels[0]
                    .lengthscales.numpy()[1]
                }
            )
            wandb.log(
                {
                    "expert_1_kernel_lengthscale_3": self.dynamics.mosvgpe.experts_list[
                        0
                    ]
                    .gp.kernel.kernels[0]
                    .lengthscales.numpy()[2]
                }
            )
            wandb.log(
                {
                    "expert_1_kernel_lengthscale_4": self.dynamics.mosvgpe.experts_list[
                        0
                    ]
                    .gp.kernel.kernels[0]
                    .lengthscales.numpy()[3]
                }
            )
            wandb.log(
                {
                    "expert_2_kernel_lengthscale_1": self.dynamics.mosvgpe.experts_list[
                        1
                    ]
                    .gp.kernel.kernels[0]
                    .lengthscales.numpy()[0]
                }
            )
            wandb.log(
                {
                    "expert_2_kernel_lengthscale_2": self.dynamics.mosvgpe.experts_list[
                        1
                    ]
                    .gp.kernel.kernels[0]
                    .lengthscales.numpy()[1]
                }
            )
            wandb.log(
                {
                    "expert_2_kernel_lengthscale_3": self.dynamics.mosvgpe.experts_list[
                        1
                    ]
                    .gp.kernel.kernels[0]
                    .lengthscales.numpy()[2]
                }
            )
            wandb.log(
                {
                    "expert_2_kernel_lengthscale_4": self.dynamics.mosvgpe.experts_list[
                        1
                    ]
                    .gp.kernel.kernels[0]
                    .lengthscales.numpy()[3]
                }
            )


def sample_inducing_inputs_from_X(
    X: InputData, num_inducing: int, active_dims: Optional[List[int]] = None
):
    idx = np.random.choice(range(X.shape[0]), size=num_inducing, replace=False)
    if isinstance(active_dims, slice):
        X = X[..., active_dims]
    elif active_dims is not None:
        X = tf.gather(X, active_dims, axis=-1)
    if isinstance(X, tf.Tensor):
        X = X.numpy()
    return X[idx, ...].reshape(-1, X.shape[1])


def mode_rl_loop(
    dynamics: ModeRLDynamics,
    env: py_environment.PyEnvironment,
    explorative_controller: ControllerInterface,
    # save_freq: Optional[Union[str, int]] = None,
    # log_dir: str = "./",
    # max_to_keep: int = None,
    callback: Callable[[], None],
    num_explorative_trajectories: int = 6,
    num_episodes: int = 30,
):
    # if isinstance(explorative_controller.start_state, tf.Tensor):
    #     start_state = explorative_controller.start_state.numpy()
    converged = False
    for i in range(1, num_episodes):

        opt_result = explorative_controller.optimise()
        callback(iteration=i)
        if converged:
            # TODO implement check for convergence
            break

        X, Y = [], []
        for j in range(num_explorative_trajectories):
            X_, Y_ = collect_data_from_env(
                env=env,
                start_state=explorative_controller.start_state,
                controls=explorative_controller(),
            )
            X.append(X_)
            Y.append(Y_)
            extrinsic_reward = explorative_controller.cost_fn(
                state=tfd.Deterministic(X_[:, : dynamics.state_dim]),
                control=tfd.Deterministic(X_[:, dynamics.state_dim :]),
            )
            # TODO add final state??
            wandb.log({"Extrinsic reward rollout " + str(j): extrinsic_reward})

        X = np.concatenate(X, 0)
        Y = np.concatenate(Y, 0)
        new_dataset = (X, Y)
        # print("new_dataset")
        # print(new_dataset)

        num_constraint_violations = 0.0
        for test_state in X[:, : dynamics.state_dim]:
            pixel = env.state_to_pixel(test_state)
            gating_value = env.gating_bitmap[pixel[0], pixel[1]]
            if gating_value < 0.5:
                num_constraint_violations += 1.0
        wandb.log({"Number constraint violations": num_constraint_violations})

        # new_data = self.explore_env()
        dynamics.update_dataset(new_dataset)
        dynamics.optimise()


@hydra.main(config_path="configs", config_name="main")
def run_experiment(cfg: omegaconf.DictConfig):
    tf.keras.utils.set_random_seed(cfg.random_seed)
    # np.random.seed(cfg.random_seed)
    ###### Initialise WandB run and save experiment config ######
    run = wandb.init(
        entity=cfg.wandb.entity,
        project=cfg.wandb.project,
        config=omegaconf.OmegaConf.to_container(
            cfg, resolve=True, throw_on_missing=True
        ),
    )
    log_dir = run.dir

    ###### Configure environment ######
    env = simenvs.make(cfg.env.name)

    # start_state = tf.constant(cfg.start_state, dtype=default_float(), shape=(1, 2))
    # target_state = tf.constant(cfg.target_state, dtype=default_float(), shape=(1, 2))
    start_state = hydra.utils.instantiate(cfg.start_state)
    target_state = hydra.utils.instantiate(cfg.target_state)

    ###### Sample initial data set from env and train dynamics on it ######
    initial_dataset = sample_env_trajectories(
        env=env,
        start_state=start_state,
        horizon=cfg.initial_dataset.horizon,
        num_trajectories=cfg.initial_dataset.num_trajectories,
        width=cfg.initial_dataset.width,
        random_seed=cfg.random_seed,
    )

    ###### Instantiate dynamics model and sample inducing inputs data ######
    # dynamics = hydra.utils.instantiate(cfg.dynamics)
    load_dir = None
    # load_dir = "./wandb/run-20221001_195555-2k2vmkt2/files/saved-models/dynamics-after-training-on-dataset-0-config.json"
    # load_dir = "./wandb/run-20221006_141321-kaquc615/files/saved-models/dynamics-after-training-on-dataset-0-config.json"
    # load_dir = "./wandb/run-20221007_151826-1pzhiyxb/files/saved-models/dynamics-after-training-on-dataset-0-config.json"
    # load_dir = "./wandb/run-20221007_154315-1fzua0ff/files/saved-models/dynamics-after-training-on-dataset-0-config.json"
    # load_dir = "./wandb/run-20221007_175642-4frzrfrz/files/saved-models/dynamics-after-training-on-dataset-1-config.json"

    if load_dir is not None:
        ###### Try to load trained dynamics model  ######
        dynamics = ModeRLDynamics.load(load_dir)
        dynamics.callbacks = dynamics_callbacks(
            dynamics, logging_epoch_freq=cfg.logging_epoch_freq
        )
        gpf.utilities.set_trainable(dynamics.mosvgpe.gating_network.gp.kernel, False)
        gpf.utilities.set_trainable(
            dynamics.mosvgpe.experts_list[1].gp.likelihood, False
        )  # Needed to stop inf in bound due to expert 2 learning very low noise variance
    else:
        dynamics = model_from_DictConfig(
            cfg.dynamics, custom_objects={"ModeRLDynamics": ModeRLDynamics}
        )
        print("dynamics")
        print(dynamics)
        # dynamics.dynamics_fit_kwargs.update({"epochs": initial_num_epochs, "batch_size": batch_size, "validation_split": 0.2})
        sample_mosvgpe_inducing_inputs_from_data(
            model=dynamics.mosvgpe, X=initial_dataset[0]
        )
        gpf.utilities.set_trainable(dynamics.mosvgpe.gating_network.gp.kernel, False)
        gpf.utilities.set_trainable(
            dynamics.mosvgpe.experts_list[1].gp.likelihood, False
        )  # Needed to stop inf in bound due to expert 2 learning very low noise variance
        # gpf.utilities.print_summary(dynamics)

        ###### Build the dynamics model ######
        dynamics.mosvgpe(initial_dataset[0])
        dynamics.desired_mode = set_desired_mode(dynamics)
        dynamics(initial_dataset[0])
        dynamics.save(
            os.path.join(log_dir, "saved-models/dynamics-before-training-config.json")
        )

        ###### Train dynamics on initial_dataset and update desired mode ######
        dynamics.update_dataset(initial_dataset)
        dynamics.callbacks = dynamics_callbacks(
            dynamics, logging_epoch_freq=cfg.logging_epoch_freq
        )
        dynamics.optimise()

        # ###### Set the desired mode and save ######
        dynamics.desired_mode = set_desired_mode(dynamics)
        dynamics.save(
            os.path.join(
                log_dir, "saved-models/dynamics-after-training-on-dataset-0-config.json"
            )
        )

    # load_dir = "./wandb/run-20221004_221209-ldmc9tcg/files/saved-models/controller-optimised-1-config.json"
    # load_dir = "./wandb/run-20221004_225725-3126jpv6/files/saved-models/controller-optimised-1-config.json"
    # explorative_controller.load(load_dir)

    gpf.utilities.print_summary(dynamics.mosvgpe.gating_network.gp.kernel)
    # dynamics.mosvgpe.gating_network.gp.kernel.lengthscales = (
    #     dynamics.mosvgpe.gating_network.gp.kernel.lengthscales * 2
    # )
    # dynamics.mosvgpe.num_data = None
    # gpf.utilities.set_trainable(dynamics.mosvgpe.gating_network.gp.kernel, True)
    # gpf.utilities.set_trainable(dynamics.mosvgpe.experts_list[1].gp.kernel, False)

    # dynamics.desired_mode = set_desired_mode(dynamics)
    if (
        "joint_gating_function_entropy"
        in cfg.explorative_controller.explorative_objective_fn
    ):
        explorative_objective_fn = joint_gating_function_entropy
    elif "bald" in cfg.explorative_controller.explorative_objective_fn:
        explorative_objective_fn = bald_objective
    elif (
        "independent_gating_function_entropy"
        in cfg.explorative_controller.explorative_objective_fn
    ):
        explorative_objective_fn = independent_gating_function_entropy
    elif "none" in cfg.explorative_controller.explorative_objective_fn:
        explorative_objective_fn = lambda *args, **kwargs: 0.0
    ###### Build greedy cost function ######
    cost_fn = hydra.utils.instantiate(cfg.cost_fn)
    ###### Configure the explorative controller (wraps cost_fn in the explorative objective) ######
    explorative_controller = ExplorativeController(
        start_state=start_state,
        dynamics=dynamics,
        explorative_objective_fn=explorative_objective_fn,
        cost_fn=cost_fn,
        control_dim=env.action_spec().shape[0],
        horizon=cfg.explorative_controller.horizon,
        max_iterations=cfg.explorative_controller.max_iterations,
        mode_satisfaction_prob=cfg.explorative_controller.mode_satisfaction_prob,
        exploration_weight=cfg.explorative_controller.exploration_weight,
        keep_last_solution=cfg.explorative_controller.keep_last_solution,
        callback=None,
        control_lower_bound=cfg.explorative_controller.control_lower_bound,
        control_upper_bound=cfg.explorative_controller.control_upper_bound,
        method=cfg.explorative_controller.method,
    )
    explorative_controller.save(
        os.path.join(log_dir, "saved-models/controller-initial-config.json")
    )

    # explorative_controller.callback = build_controller_plotting_callback(
    #     env=env,
    #     controller=explorative_controller,
    #     target_state=target_state,
    #     logging_epoch_freq=10,
    # )
    # explorative_controller.optimise()

    # Plot final trajectory
    test_inputs = create_test_inputs(40000)
    # fig = plot_trajectories_over_desired_mixing_prob(
    #     env,
    #     controller=explorative_controller,
    #     test_inputs=test_inputs,
    #     target_state=target_state,
    # )
    # wandb.log({"Final traj over desired mixing prob": wandb.Image(fig)})
    # fig = plot_trajectories_over_desired_gating_gp(
    #     env,
    #     controller=explorative_controller,
    #     test_inputs=test_inputs,
    #     target_state=target_state,
    # )
    # wandb.log({"Final traj over desired gating gp": wandb.Image(fig)})
    def callback(iteration: int):
        # print("before callback")
        # gpf.utilities.print_summary(dynamics.mosvgpe)
        print("initial solution")
        gpf.utilities.print_summary(
            explorative_controller.trajectory_optimiser.initial_solution
        )
        gpf.utilities.print_summary(
            explorative_controller.trajectory_optimiser.previous_solution
        )
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
        explorative_controller.save(
            os.path.join(
                log_dir,
                "saved-models/controller-optimised-{}-config.json".format(iteration),
            )
        )
        dynamics.save(
            os.path.join(
                log_dir,
                "saved-models/dynamics-after-training-on-dataset-{}-config.json".format(
                    iteration
                ),
            )
        )

    # Run the mbrl loop
    mode_rl_loop(
        dynamics=dynamics,
        env=env,
        explorative_controller=explorative_controller,
        callback=callback,
        num_explorative_trajectories=cfg.num_explorative_trajectories,
        num_episodes=cfg.num_episodes,
    )


if __name__ == "__main__":
    run_experiment()  # pyright: ignore
