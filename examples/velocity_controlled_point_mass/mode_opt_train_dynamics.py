#!/usr/bin/env python3
from typing import List, Optional

import gpflow as gpf
import hydra
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from modeopt.custom_types import Dataset
from modeopt.utils import model_from_DictConfig
from modeopt.mode_opt import ModeOpt
from mogpe.keras.callbacks.tensorboard import PlotFn, TensorboardImageCallback
from mogpe.keras.mixture_of_experts import MixtureOfSVGPExperts
from mogpe.keras.plotting import MixtureOfSVGPExpertsContourPlotter
from mogpe.keras.utils import sample_mosvgpe_inducing_inputs_from_data
from omegaconf import DictConfig

from velocity_controlled_point_mass.data.utils import load_vcpm_dataset

tfd = tfp.distributions

meaning_of_life = 42
tf.random.set_seed(meaning_of_life)
np.random.seed(meaning_of_life)

# @hydra.main(
#     config_path="keras_configs/scenario_5", config_name="mode_opt_train_dynamics"
# )
@hydra.main(
    config_path="keras_configs/scenario_5/train_dynamics",
    # config_path="keras_configs/scenario_7/train_dynamics",
    config_name="mode_opt_train_dynamics"
    # config_name="mode_opt_train_dynamics_const_action",
)
def mode_opt_train_dynamics_from_cfg(cfg: DictConfig):
    mode_optimiser = model_from_DictConfig(
        cfg.mode_opt, custom_objects={"ModeOpt": ModeOpt}
    )

    dataset, _ = load_vcpm_dataset(
        filename=cfg.dataset.filename,
        trim_coords=cfg.dataset.trim_coords,
        standardise=cfg.dataset.standardise,
    )
    mode_optimiser.dataset = dataset
    sample_mosvgpe_inducing_inputs_from_data(
        dataset[0], mode_optimiser.dynamics.mosvgpe
    )
    # gpf.utilities.set_trainable(
    #     mode_optimiser.dynamics.mosvgpe.gating_network.gp.inducing_variable, False
    # )
    gpf.utilities.print_summary(
        mode_optimiser.dynamics.mosvgpe.gating_network.gp.inducing_variable
    )

    plotting_callbacks = build_contour_plotter_callbacks(
        mode_optimiser.dynamics.mosvgpe,
        dataset=dataset,
        logging_epoch_freq=cfg.logging_epoch_freq,
    )
    mode_optimiser.add_dynamics_callbacks(plotting_callbacks)
    mode_optimiser(dataset[0])
    mode_optimiser.save("./ckpts/ModeOpt")
    mode_optimiser.optimise_dynamics()


# @hydra.main(config_path="keras_configs/scenario_5", config_name="train_dynamics")
# def train_dynamics_from_cfg(cfg: DictConfig):
#     train_dataset, _ = load_vcpm_dataset(
#         filename=cfg.dataset.filename,
#         trim_coords=cfg.dataset.trim_coords,
#         standardise=cfg.dataset.standardise,
#     )
#     X, Y = train_dataset

#     dynamics = model_from_DictConfig(
#         cfg.dynamics, custom_objects={"ModeOptDynamics": ModeOptDynamics}
#     )
#     mode_optimiser = ModeOpt(
#         start_state=[3.0, -1.0],
#         target_state=[-3.0, 2.5],
#         # dataset=dataset,
#         env_name="velocity-controlled-point-mass/scenario-5",
#         dynamics=dynamics,
#     )
#     json_cfg = tf.keras.utils.serialize_keras_object(mode_optimiser)
#     print(type(json_cfg))
#     print(json_cfg)
#     json_cfg = tf.keras.layers.serialize(mode_optimiser)
#     import yaml
#     import json

#     print(type(json_cfg))
#     print(json_cfg)
#     # with open("./mode_cfg.yaml", "w") as yaml_file:
#     #     yaml.dump(json_cfg, yaml_file, allow_unicode=True)
#     with open("./mode_cfg.json", "w") as file:
#         json_cfg = json.dumps(json_cfg, ensure_ascii=False)
#         file.write(json_cfg)
#         # json.dump(json_cfg, file, encoding="utf-8")


def build_contour_plotter_callbacks(
    mosvgpe: MixtureOfSVGPExperts,
    dataset: Dataset,
    logging_epoch_freq: Optional[int] = 30,
    log_dir: Optional[str] = "./logs",
) -> List[PlotFn]:
    mosvgpe_plotter = MixtureOfSVGPExpertsContourPlotter(mosvgpe, dataset=dataset)
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
    # train_dynamics_from_cfg()
    mode_opt_train_dynamics_from_cfg()
