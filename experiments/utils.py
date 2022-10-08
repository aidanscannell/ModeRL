#!/usr/bin/env python3
import json
from typing import List, Optional

import mosvgpe
import numpy as np
import tensorflow as tf
from gpflow.inducing_variables import InducingVariables
from moderl.custom_types import Dataset, State
from moderl.dynamics.dynamics import ModeRLDynamics
from moderl.rollouts import collect_data_from_env
from mosvgpe.custom_types import InputData
from mosvgpe.experts import SVGPExpert
from mosvgpe.mixture_of_experts import MixtureOfSVGPExperts
from omegaconf import DictConfig, OmegaConf


class TwoExpertsList(list):
    def __init__(self, expert_one: SVGPExpert, expert_two: SVGPExpert):
        super().__init__()
        self.append(expert_one)
        self.append(expert_two)


def sample_env_trajectories(
    env,
    start_state: State,
    horizon: int = 15,
    num_trajectories: int = 10,
    width: float = 1.0,
    random_seed: int = 42,
) -> Dataset:
    np.random.seed(random_seed)

    X, Y = [], []
    for i in range(num_trajectories):
        controls = np.random.uniform(
            env.action_spec().minimum,
            env.action_spec().maximum,
            (horizon, env.action_spec().shape[-1]),
        )
        X_, Y_ = collect_data_from_env(env, start_state=start_state, controls=controls)
        X.append(X_)
        Y.append(Y_)
    X = np.concatenate(X, 0)
    Y = np.concatenate(Y, 0)
    mask_1 = X[:, 0] > start_state[0, 0] - width
    mask_2 = X[:, 0] < start_state[0, 0] + width
    mask_3 = X[:, 1] > start_state[0, 1] - width
    mask_4 = X[:, 1] < start_state[0, 1] + width
    mask = mask_1 & mask_2 & mask_3 & mask_4
    X_trimmed = X[mask]
    Y_trimmed = Y[mask]
    # print(X_trimmed.shape)
    # print(Y_trimmed.shape)
    return (X_trimmed, Y_trimmed)


def model_from_DictConfig(
    cfg: DictConfig, custom_objects: dict = None
) -> ModeRLDynamics:
    return tf.keras.models.model_from_json(
        json.dumps(OmegaConf.to_container(cfg)), custom_objects=custom_objects
    )


def sample_mosvgpe_inducing_inputs_from_data(model: MixtureOfSVGPExperts, X: InputData):
    # TODO should inducing inputs only be for active dims or all inputs?
    for expert in model.experts_list:
        sample_inducing_variables_from_data(X, expert.gp.inducing_variable)
    sample_inducing_variables_from_data(
        X,
        model.gating_network.gp.inducing_variable,
        # active_dims=model.gating_network.gp.kernel.active_dims,
    )


def sample_inducing_variables_from_data(
    X: InputData,
    inducing_variable: InducingVariables,
    active_dims: Optional[List[int]] = None,
):
    if isinstance(
        inducing_variable,
        mosvgpe.keras.inducing_variables.SharedIndependentInducingVariablesSerializable,
    ):
        inducing_variable.inducing_variable.Z.assign(
            sample_inducing_inputs_from_data(
                X,
                inducing_variable.inducing_variable.Z.shape[0],
                active_dims=active_dims,
            )
        )
    elif isinstance(
        inducing_variable,
        mosvgpe.keras.inducing_variables.SeparateIndependentInducingVariablesSerializable,
    ):
        for inducing_var in inducing_variable.inducing_variables:
            Z = sample_mosvgpe_inducing_inputs_from_data(X, inducing_var.Z)
            inducing_var.Z.assign(Z)
    else:
        inducing_variable.Z.assign(
            sample_inducing_inputs_from_data(
                X, inducing_variable.Z.shape[0], active_dims=active_dims
            )
        )


def sample_inducing_inputs_from_data(
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
