#!/usr/bin/env python3
from typing import List, Optional

import gpflow as gpf
import hydra
import mlflow
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from modeopt.custom_types import Dataset
from modeopt.mode_opt import ModeOpt
from modeopt.utils import log_params_from_omegaconf_dict, model_from_DictConfig
from mogpe.keras.callbacks.tensorboard import PlotFn, TensorboardImageCallback
from mogpe.keras.mixture_of_experts import MixtureOfSVGPExperts
from mogpe.keras.plotting import MixtureOfSVGPExpertsContourPlotter
from mogpe.keras.utils import sample_mosvgpe_inducing_inputs_from_data
from omegaconf import DictConfig

from velocity_controlled_point_mass.data.utils import load_vcpm_dataset
from velocity_controlled_point_mass.mode_opt_riemannian_energy_traj_opt import (
    create_test_inputs,
)

tfd = tfp.distributions

meaning_of_life = 42
tf.random.set_seed(meaning_of_life)
np.random.seed(meaning_of_life)


@hydra.main(
    # config_path="keras_configs/scenario_5/train_dynamics",
    config_path="keras_configs/scenario_10/train_dynamics",
    # config_path="keras_configs/scenario_7/train_dynamics",
    # config_path="keras_configs/scenario_8/train_dynamics",
    # config_path="keras_configs/scenario_9/train_dynamics",
    # config_name="mode_opt_train_dynamics",
    config_name="mode_opt_train_dynamics_initial",
)
def mode_opt_train_dynamics_from_cfg(cfg: DictConfig):
    # Set path to mlruns directory
    # mlflow.set_tracking_uri("file://" + hydra.utils.get_original_cwd() + "/mlruns")
    # mlflow.set_experiment(cfg.mlflow.experiment_name)

    # with mlflow.start_run():
    #     log_params_from_omegaconf_dict(cfg)
    #     mlflow.keras.autolog()

    mode_optimiser = model_from_DictConfig(
        cfg.mode_opt, custom_objects={"ModeOpt": ModeOpt}
    )
    # mode_optimiser.dynamics.mosvgpe._bound = "tight"
    # print("mode_optimiser.dynamics.mosvgpe.bound")
    # print(mode_optimiser.dynamics.mosvgpe.bound)

    dataset, _ = load_vcpm_dataset(
        filename=cfg.dataset.filename,
        trim_coords=cfg.dataset.trim_coords,
        standardise=cfg.dataset.standardise,
        # plot=True,
    )
    mode_optimiser.dataset = dataset
    sample_mosvgpe_inducing_inputs_from_data(
        dataset[0], mode_optimiser.dynamics.mosvgpe
    )
    # gpf.utilities.set_trainable(
    #     mode_optimiser.dynamics.mosvgpe.gating_network.gp.inducing_variable, False
    # )
    # gpf.utilities.print_summary(
    #     mode_optimiser.dynamics.mosvgpe.gating_network.gp.inducing_variable
    # )

    plotting_callbacks = build_contour_plotter_callbacks(
        mode_optimiser.dynamics.mosvgpe,
        dataset=dataset,
        logging_epoch_freq=cfg.logging_epoch_freq,
    )
    mode_optimiser.add_dynamics_callbacks(plotting_callbacks)

    mode_optimiser.optimise_dynamics()


def build_contour_plotter_callbacks(
    mosvgpe: MixtureOfSVGPExperts,
    dataset: Dataset,
    logging_epoch_freq: Optional[int] = 30,
    log_dir: Optional[str] = "./logs",
) -> List[PlotFn]:
    test_inputs = create_test_inputs(x_min=[-3, -3], x_max=[3, 3], input_dim=4)
    # mosvgpe_plotter = MixtureOfSVGPExpertsContourPlotter(mosvgpe, dataset=dataset)
    mosvgpe_plotter = MixtureOfSVGPExpertsContourPlotter(
        mosvgpe, test_inputs=test_inputs
    )
    experts_plotting_cb = TensorboardImageCallback(
        plot_fn=mosvgpe_plotter.plot_experts_gps,
        logging_epoch_freq=logging_epoch_freq,
        log_dir=log_dir,
        name="Experts' latent function GPs",
    )
    gating_gps_plotting_cb = TensorboardImageCallback(
        plot_fn=mosvgpe_plotter.plot_gating_network_gps,
        logging_epoch_freq=logging_epoch_freq,
        log_dir=log_dir,
        name="Gating function GPs",
    )
    mixing_probs_plotting_cb = TensorboardImageCallback(
        plot_fn=mosvgpe_plotter.plot_mixing_probs,
        logging_epoch_freq=logging_epoch_freq,
        log_dir=log_dir,
        name="Mixing probabilities",
    )
    return [experts_plotting_cb, mixing_probs_plotting_cb, gating_gps_plotting_cb]


if __name__ == "__main__":
    mode_opt_train_dynamics_from_cfg()
