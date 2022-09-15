import hydra
import numpy as np
import omegaconf
import wandb
import tensorflow as tf
from simenvs.core import make


@hydra.main(config_path="configs", config_name="main")
def run_mode_rl(cfg: omegaconf.DictConfig):
    """Run ModeRL form a yaml config

    This function parses an yaml experiment config and runs the
    main mode_opt loop whilst also providing WandB experiment tracking.

    - Saves the experiment config
    - Saves initial data set sampled from environment
    - Monitors dynamics training
    - Monitors explorative trajectory optimisation
    """

    wandb.config = omegaconf.OmegaConf.to_container(
        cfg, resolve=True, throw_on_missing=True
    )
    run = wandb.init(entity=cfg.wandb.entity, project=cfg.wandb.project)

    start_state = tf.reshape(
        tf.constant(cfg.start_state, dtype=default_float()), shape=(1, -1)
    )
    target_state = tf.reshape(
        tf.constant(cfg.target_state, dtype=default_float()), shape=(1, -1)
    )

    explorative_controller = ExplorativeController(
        start_state,
        target_state,
        dynamics,
        cfg.dynamics.desired_mode,
        horizon=cfg.explorative_controller.horizon,
        control_dim=cfg.dynamics.control_dim,
        max_iterations=cfg.explorative_controller.max_iterations,
        mode_satisfaction_prob=cfg.mode_satisfaction_probability,
        terminal_state_cost_weight=cfg.terminal_state_cost_weight,
        state_diff_cost_weight=cfg.state_diff_cost_weight,
        control_cost_weight=cfg.control_cost_weight,
        # keep_last_solution
        method=cfg.explorative_controller.method,
    )

    # dynamics =
    mode_controller = None

    mode_opt_loop(
        start_state=start_state,
        target_state=target_state,
        dynamics=dynamics,
        mode_controller=mode_controller,
        env_name=cfg.env_name,
        explorative_controller=explorative_controller,
        dataset=initial_dataset,
        desired_mode=cfg.desired_mode,
        mode_satisfaction_probability=cfg.mode_satisfaction_probability,
        learning_rate=cfg.learning_rate,
        epsilon=cfg.epsilon,
        save_freq=cfg.save_freq,
        log_dir=cfg.log_dir,
        dynamics_fit_kwargs=cfg.dynamics_fit_kwargs,
        max_to_keep=cfg.max_to_keep,
        num_explorative_trajectories=cfg.num_explorative_trajectories,
    )


def build_callbacks():
    self._log_dir = os.path.join(
        log_dir, datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")
    )
    self.dynamics_callbacks = [
        tf.keras.callbacks.TensorBoard(log_dir=os.path.join(self._log_dir, "logs"))
    ]
    if self.save_freq is not None:
        self.dynamics_callbacks.append(
            tf.keras.callbacks.ModelCheckpoint(
                filepath=os.path.join(self._log_dir, "ckpts/ModeOptDynamics"),
                monitor="loss",
                save_format="tf",
                save_best_only=True,
                save_freq=self.save_freq,
            )
        )

    self.ckpt_dir = os.path.join(self._log_dir, "ckpts")
    self.create_checkpoint_manager()


# @hydra.main(config_path="configs", config_name="main")
# def run_mode_opt(cfg: omegaconf.DictConfig):
#     wandb.config = omegaconf.OmegaConf.to_container(
#         cfg, resolve=True, throw_on_missing=True
#     )
#     run = wandb.init(entity=cfg.wandb.entity, project=cfg.wandb.project)
#     wandb.log({"loss": loss})
#     model = Model(**wandb.config.model.configs)

#     run = wandb.init(
#         project="sb3",
#         config=config,
#         sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
#         monitor_gym=True,  # auto-upload the videos of agents playing the game
#         save_code=True,  # optional
#     )

#     env = make(cfg.env_name)

#     wandb.log({"reward": env.reward})
#     wandb.log(
#         {
#             "Value Function": wandb.plots.HeatMap(
#                 list(range(environment.height)),
#                 list(range(environment.width)),
#                 value_map,
#             )
#         }
#     )

#     np.random.seed(cfg.seed)
#     tf.random.set_seed(cfg.seed)
#     # if cfg.algorithm.name == "pets":
#     # return pets.train(env, term_fn, reward_fn, cfg)
#     # if cfg.algorithm.name == "mbpo":
#     # test_env, *_ = mbrl.util.env.EnvHandler.make_env(cfg)
#     # return mbpo.train(env, test_env, term_fn, cfg)
#     # if cfg.algorithm.name == "planet":
#     # return planet.train(env, cfg)


# def make_env():
#     env = gym.make(config["env_name"])
#     env = Monitor(env)  # record stats such as returns

#     return env


# def build_mode_opt():

#     mode_opt = ModeOptimiser(
#         start_state,
#         target_state,
#         dynamics,
#         mode_controller,
#         env_name,
#         explorative_controller,
#         dataset,
#         desired_mode,
#         mode_satisfaction_probability,
#         learning_rate,
#         epsilon,
#         save_freq,
#         log_dir,
#         dynamics_fit_kwargs,
#         max_to_keep,
#         num_explorative_trajectories,
#     )


# def build_dynamics_model(num_experts: int = 2):

#     dynamics = ModeOptDynamics()
#     desired_mode = 1
#     experts_list = [buld_svgp_expert() for _ in range(num_experts)]
#     svgp_gating_network = buld_svgp_gating_network(lengthscales, variance)
#     mosvgpe = MixtureOfSVGPExperts(
#         experts_list=experts_list, gating_network=svgp_gating_network
#     )


# def build_svgp_expert(
#     output_dim: int = 2,
#     mean_function_const=[0.0],
#     lengthscales=[1.0],
#     noise_variance=[1.0],
#     num_inducing: int = 90,
# ):
#     likelihood = Gaussian(variance=noise_variance * output_dim)
#     kernel = SeparateIndependent(kernel_list)
#     mean_function = Constant(mean_function_const * output_dim)
#     inducing_variable = SharedIndependentInducingVariables(
#         build_inducing_points(X, num_inducing)
#     )
#     expert = SVGPExpert(
#         kernel=kernel,
#         likelihood=likelihood,
#         inducing_variable=inducing_variable,
#         num_latent_gps=output_dim,
#     )


# def build_svgp_gating_network(
#     legnthscales: List[float] = [1.0, 1.0], variance: float = 1.0
# ):
#     kernel = SquaredExponential(
#         lengthscales=tf.Variable(lengthscales, dtype=default_float()),
#         variance=tf.Variable(variance, dtype=default_float()),
#     )
#     SVGPGatingNetwork(
#         kernel=kernel,
#         inducing_variable=inducing_variable,
#         # q_diag: bool = False,
#         q_mu=q_mu,
#         q_sqrt=q_sqrt,
#         whiten=whiten,
#     )

#     # gating_network= SVGPGatingNetwork(
#     #             kernel=SquaredExponential
#     #               config:
#     #                 lengthscales: [1.0, 1.0]
#     #                 variance: 1.0
#     #                 active_dims: [0, 1]
#     #             inducing_variable:
#     #               class_name: InducingPoints
#     #               config:
#     #                 num_inducing: 100
#     #                 input_dim: 2
#     #     predict_state_difference: true
#     #     state_dim: 2


# def build_multi_output_squared_exponential_kernel(
#     output_dim: int = 2, lengthscales=[1.0], variance=1.0
# ):
#     kernel_list = []
#     for _ in range(output_dim):
#         lengthscales = tf.Variable(lengthscales, dtype=default_float())
#         variance = tf.Variable(variance, dtype=default_float())
#         kernel_list.append(SquaredExponential(lengthscales=lengthscales))
#     kernel = SeparateIndependent(kernel_list)
#     return kernel


# def build_inducing_points(X: InputData, num_inducing: int):
#     num_data, input_dim = X.shape
#     idx = np.random.choice(range(num_data), size=num_inducing, replace=False)
#     if isinstance(X, tf.Tensor):
#         X = X.numpy()
#     inducing_variable = X[idx, ...].reshape(-1, input_dim)
#     return gpf.inducing_variables.InducingPoints(inducing_variable)


# if __name__ == "__main__":
#     run_mode_opt()


# # #!/usr/bin/env python3
# # from typing import List, Optional

# # import tensorflow as tf
# # import tensorflow_probability as tfp
# # from gpflow import default_float
# # from modeopt.dynamics import ModeOptDynamics
# # from modeopt.controllers.utils import build_riemannian_energy_controller
# # from modeopt.controllers.non_feedback.explorative_controller import (
# #     ExplorativeController,
# # )
# # from modeopt.mode_opt import ModeOpt
# # from modeopt.monitor.callbacks import PlotFn, TensorboardImageCallbackScipy
# # from modeopt.plotting import ModeOptContourPlotter
# # from omegaconf import DictConfig

# # tfd = tfp.distributions

# # def mode_opt_explore_from_cfg(cfg: DictConfig):
# #     dynamics = tf.keras.models.load_model(
# #         cfg.dynamics.ckpt_dir, custom_objects={"ModeOptDynamics": ModeOptDynamics}
# #     )
# #     dynamics.desired_mode = cfg.dynamics.desired_mode  # update desired mode
# #     start_state = tf.reshape(
# #         tf.constant(cfg.start_state, dtype=default_float()), shape=(1, -1)
# #     )
# #     target_state = tf.reshape(
# #         tf.constant(cfg.target_state, dtype=default_float()), shape=(1, -1)
# #     )
# #     terminal_state_cost_matrix = tf.constant(
# #         cfg.cost_fn.terminal_state_cost_matrix, dtype=default_float()
# #     )
# #     control_cost_matrix = tf.constant(
# #         cfg.cost_fn.control_cost_matrix, dtype=default_float()
# #     )

# #     explorative_controller = ExplorativeController(
# #         start_state,
# #         target_state,
# #         dynamics,
# #         cfg.dynamics.desired_mode,
# #         horizon=cfg.horizon,
# #         control_dim=cfg.dynamics.control_dim,
# #         max_iterations=cfg.max_iterations,
# #         mode_satisfaction_prob=cfg.mode_satisfaction_prob,
# #         terminal_state_cost_weight=cfg.terminal_state_cost_weight,
# #         state_diff_cost_weight=cfg.state_diff_cost_weight,
# #         control_cost_weight=cfg.control_cost_weight,
# #         # keep_last_solution
# #         method=cfg.method,
# #     )

# #     mode_optimiser = ModeOpt(
# #         start_state,
# #         target_state,
# #         env_name=cfg.env_name,
# #         dynamics=dynamics,
# #         explorative_controller=explorative_controller,
# #     )

# #     plotting_callbacks = build_contour_plotter_callbacks(
# #         mode_optimiser, logging_freq=cfg.logging_freq
# #     )
# #     mode_optimiser.mode_controller_callback = plotting_callbacks
# #     mode_optimiser.optimise_mode_controller()


# # def build_contour_plotter_callbacks(
# #     mode_optimiser: ModeOpt,
# #     logging_freq: Optional[int] = 3,
# #     log_dir: Optional[str] = "./logs",
# # ) -> List[PlotFn]:
# #     test_inputs = create_test_inputs(x_min=[-3, -3], x_max=[3, 3], input_dim=4)

# #     mode_optimiser_plotter = ModeOptContourPlotter(
# #         mode_optimiser=mode_optimiser, test_inputs=test_inputs
# #     )
# #     gating_gps_plotting_cb = TensorboardImageCallbackScipy(
# #         plot_fn=mode_optimiser_plotter.plot_trajectories_over_gating_network_gps,
# #         logging_epoch_freq=logging_freq,
# #         log_dir=log_dir,
# #         name="Trajectories over gating function GPs",
# #     )
# #     mixing_probs_plotting_cb = TensorboardImageCallbackScipy(
# #         plot_fn=mode_optimiser_plotter.plot_trajectories_over_mixing_probs,
# #         logging_epoch_freq=logging_freq,
# #         log_dir=log_dir,
# #         name="Trajectories over mixing probabilities",
# #     )

# #     def plotting_callback(step, variables, value):
# #         mixing_probs_plotting_cb(step, variables, value)
# #         gating_gps_plotting_cb(step, variables, value)

# #     return plotting_callback
