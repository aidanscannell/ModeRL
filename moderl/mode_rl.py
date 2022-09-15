#!/usr/bin/env python3
import datetime
import os
from typing import Callable, List, Optional, Union

from tf_agents.environments import py_environment
import numpy as np
import simenvs
import tensor_annotations.tensorflow as ttf
import tensorflow as tf
from gpflow import default_float

# from gpflow.utilities.keras import try_array_except_none, try_val_except_none
from tensor_annotations.axes import Batch

from moderl.controllers import (
    CONTROLLER_OBJECTS,
    FeedbackController,
    NonFeedbackController,
)
from moderl.custom_types import Dataset, StateDim
from moderl.dynamics import ModeRLDynamics
from moderl.rollouts import (
    collect_data_from_env,
    rollout_controller_in_dynamics,
    rollout_controller_in_env,
)

Callback = Callable
Controller = Union[FeedbackController, NonFeedbackController]

DEFAULT_DYNAMICS_FIT_KWARGS = {
    "batch_size": 16,
    "epochs": 1000,
    "verbose": True,
    "validation_split": 0.2,
}


def mode_rl(
    start_state: ttf.Tensor2[Batch, StateDim],
    target_state: ttf.Tensor2[Batch, StateDim],
    dynamics: ModeRLDynamics,
    mode_controller: Controller,
    env: py_environment.PyEnvironment,
    explorative_controller: Controller = None,
    initial_dataset: Dataset = None,
    desired_mode: int = 1,
    mode_satisfaction_probability: float = 0.7,  # Mode satisfaction probability (0, 1]
    learning_rate: float = 0.01,
    epsilon: float = 1e-8,
    save_freq: Optional[Union[str, int]] = None,
    log_dir: str = "./",
    dynamics_fit_kwargs: dict = DEFAULT_DYNAMICS_FIT_KWARGS,
    max_to_keep: int = None,
    num_explorative_trajectories: int = 6,
    dynamics_callbacks=None,
    explorative_controller_callback=None,
    mode_controller_callback=None,
):
    at_target_state = False
    while not at_target_state:

        opt_result = explorative_controller.optimise(explorative_controller_callback)
        X, Y = [], []
        for i in range(num_explorative_trajectories):
            if isinstance(start_state, tf.Tensor):
                start_state = start_state.numpy()
            X_, Y_ = collect_data_from_env(
                env=env,
                start_state=start_state,
                controls=explorative_controller(),
            )
            X.append(X_)
            Y.append(Y_)
        X = np.concatenate(X, 0)
        Y = np.concatenate(Y, 0)

        # new_data = self.explore_env()
        update_dataset(new_data=new_data)

        # Optimise Dynamics
        dynamics.mosvgpe._num_data = dataset[0].shape[0]
        dynamics(dataset[0])  # Needs to be called to build shapes
        dynamics.mosvgpe(dataset[0])  # Needs to be called to build shapes
        # TODO: if callbacks in self.dynamics_fit_kwargs extract and append them
        dynamics.fit(
            dataset[0], dataset[1], callbacks=dynamics_callbacks, **dynamics_fit_kwargs
        )

        (trajectory, at_target_state) = self.find_trajectory_to_target()
        if at_target_state:
            in_desired_mode = self.check_mode_remaining(trajectory)
            if in_desired_mode:
                print("Found delta mode remaining trajectory to target state")
                return True
            else:
                at_target_state = False

    print("opt_result['success']")
    print(opt_result["success"])
    if not opt_result["success"]:
        raise NotImplementedError
        # explorative_controller.reset()
        # return explore_env()


def update_dataset(new_dataset: Dataset, dataset: Dataset = None):
    if dataset is not None:
        Xold, Yold = dataset
        Xnew, Ynew = new_dataset
        X = np.concatenate([Xold, Xnew], 0)
        Y = np.concatenate([Yold, Ynew], 0)
        dataset = (X, Y)
    else:
        dataset = new_data
    return dataset
