#!/usr/bin/env python3
import os
from typing import Optional, Union
from gpflow import default_float

import numpy as np
import simenvs
import tensor_annotations.tensorflow as ttf
import tensorflow as tf
from gpflow.utilities.keras import try_array_except_none, try_val_except_none
from tensor_annotations.axes import Batch

from modeopt.controllers import FeedbackController, NonFeedbackController
from modeopt.custom_types import Dataset, StateDim
from modeopt.dynamics import ModeOptDynamics
from modeopt.rollouts import (
    rollout_controller_in_dynamics,
    rollout_controller_in_env,
    rollout_controls_in_dynamics,
    rollout_controls_in_env,
)

Controller = Union[FeedbackController, NonFeedbackController]

DEFAULT_DYNAMICS_FIT_KWARGS = {
    "batch_size": 16,
    "epochs": 1000,
    "verbose": True,
    "validation_split": 0.2,
}


class ModeOpt(tf.keras.Model):
    def __init__(
        self,
        start_state: ttf.Tensor2[Batch, StateDim],
        target_state: ttf.Tensor2[Batch, StateDim],
        env_name: str,
        dynamics: ModeOptDynamics,
        mode_controller: Controller,
        # explorative_controller: Controller,
        dataset: Dataset = None,
        desired_mode: int = 1,
        mode_satisfaction_probability: float = 0.7,  # Mode satisfaction probability (0, 1]
        # batch_size: int = 16,
        # num_epochs: int = 1000,
        # validation_split: float = 0.2,
        learning_rate: float = 0.01,
        epsilon: float = 1e-8,
        save_freq: Optional[Union[str, int]] = None,
        log_dir: str = "./",
        dynamics_fit_kwargs: dict = DEFAULT_DYNAMICS_FIT_KWARGS,
        name: str = "ModeOpt",
    ):
        super().__init__(name=name)
        # TODO how to handle increasing number of inducing points?
        self.start_state = start_state
        self.target_state = target_state
        self.env_name = env_name
        self.env = simenvs.make(env_name)
        self.dynamics = dynamics
        self.dataset = dataset
        self.desired_mode = desired_mode
        self.mode_satisfaction_probability = mode_satisfaction_probability
        # self.batch_size = batch_size
        # self.num_epochs = num_epochs
        # self.validation_split = validation_split
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.save_freq = save_freq
        self.log_dir = log_dir
        self.dynamics_fit_kwargs = dynamics_fit_kwargs

        self.mode_controller = mode_controller

        # TODO add early stopping callback

        # Compile dynamics and initialise callbacks
        optimiser = tf.keras.optimizers.Adam(
            learning_rate=learning_rate, epsilon=epsilon
        )
        self.dynamics.compile(optimizer=optimiser)
        # self.dynamics(self.dataset[0])  # Needs to be called to build shapes
        self.dynamics_callbacks = [
            tf.keras.callbacks.TensorBoard(log_dir=os.path.join(log_dir + "./logs"))
        ]
        if save_freq is not None:
            # save_freq = int(initial_dataset[0].shape[0] / batch_size)
            self.dynamics_callbacks.append(
                tf.keras.callbacks.ModelCheckpoint(
                    filepath=os.path.join(log_dir + "ckpts/ModeOptDynamics"),
                    monitor="loss",
                    save_format="tf",
                    save_best_only=True,
                    save_freq=save_freq,
                )
            )

    def call(self, input, training=False):
        if not training:
            return self.dynamics(input)

    # def save(self):
    #     import json

    #     json_cfg = tf.keras.utils.serialize_keras_object(self)
    #     print(json_cfg)
    #     # with open("./ckpts/ModeOpt/config.json", "w") as file:
    #     with open("./config.json", "w") as json_file:
    #         json_cfg = json.dumps(json_cfg, ensure_ascii=False)
    # json_cfg = json.dumps(json_cfg)
    # json_file.write(json_cfg)
    # json_file.write(json_cfg)
    # json.dump(json_cfg, file, encoding="utf-8")
    # self.save(os.path.join(log_dir + "ckpts/ModeOpt"))

    def optimise(self):
        at_target_state = False
        while not at_target_state:
            new_data = self.explore_env()
            self.update_dataset(new_data=new_data)
            self.optimise_dynamics()
            (trajectory, at_target_state) = self.find_trajectory_to_target()
            if at_target_state:
                in_desired_mode = self.check_mode_remaining(trajectory)
                if in_desired_mode:
                    print("Found delta mode remaining trajectory to target state")
                    return True
                else:
                    at_target_state = False

    def optimise_dynamics(self):
        X, Y = self.dataset
        self.dynamics.mosvgpe._num_data = X.shape[0]
        self.dynamics(X)  # Needs to be called to build shapes
        self.dynamics.mosvgpe(X)  # Needs to be called to build shapes
        # TODO: if callbacks in self.dynamics_fit_kwargs extract and append them
        self.dynamics.fit(
            X,
            Y,
            callbacks=self.dynamics_callbacks,
            **self.dynamics_fit_kwargs,
            # batch_size=self.batch_size,
            # epochs=self.num_epochs,
            # verbose=self.verbose,
            # validation_split=self.validation_split,
        )
        self.save(os.path.join(self.log_dir, "ckpts/ModeOpt"))

    def update_dataset(self, new_data: Dataset):
        if self.dataset is not None:
            Xold, Yold = self.dataset
            Xnew, Ynew = new_data
            X = np.concatenate([Xold, Xnew], 0)
            Y = np.concatenate([Yold, Ynew], 0)
            self.dataset = (X, Y)
        else:
            self.dataset = new_data

    def explore_env(self) -> Dataset:
        """Optimise the controller and use it to explore the environment"""
        self.explorative_controller.optimise()
        # self.env_rollout(self.explorative_controller)
        return rollout_controller_in_env(
            env=self.env,
            controller=self.explorative_controller,
            start_state=self.start_state,
        )
        # return self.env_rollout(explorative_trajectory)

    # def optimise_controller(self):
    def mode_remaining_trajectory_optimisation(self):
        plan = self.mode_trajectory_optimiser.optimise(self.mode_objective_fn)
        return plan

        # self.controller.set_trainable(True)
        # trajectory = self.controller.optimise()
        # self.controller.set_trainable(False)
        # return trajectory

    # def check_mode_remaining(self, trajectory):
    #     mode_probs = self.dynamics.predict_mode_probability(state_mean, control_mean)
    #     if (mode_probs < self.mode_satisfaction_probability).any():
    #         return False
    #     else:
    #         return True

    def dynamics_rollout(self):
        states = self.mode_controller.previous_solution.states
        print("states")
        print(states)
        print("self.env.low_process_noise_mean")
        print(self.env.low_process_noise_mean)
        corrected_states = states - self.env.low_process_noise_mean
        print("corrected_states")
        print(corrected_states)
        diff = corrected_states[1:, :] - corrected_states[:-1, :]
        controls = tf.concat(
            [diff, tf.zeros([1, states.shape[-1]], dtype=default_float())], 0
        ) * [2.4, 9.0]
        print("controls corrected")
        print(controls)
        print("controls original")
        print(self.mode_controller.previous_solution.controls)
        return rollout_controls_in_dynamics(
            dynamics=self.dynamics,
            control_means=controls,
            control_vars=None,
            start_state=self.start_state,
        )
        # return rollout_controller_in_dynamics(
        #     dynamics=self.dynamics,
        #     controller=self.mode_controller,
        #     start_state=self.start_state,
        # )

    def env_rollout(self) -> Dataset:
        return rollout_controller_in_env(
            env=self.env, controller=self.mode_controller, start_state=self.start_state
        )

    def add_dynamics_callbacks(self, callbacks):
        self.dynamics_callbacks.append(callbacks)

    @property
    def dataset(self):
        return self._dataset

    @dataset.setter
    def dataset(self, dataset: Dataset):
        if dataset is not None:
            self.dynamics.mosvgpe._num_data = dataset[0].shape[0]
        self._dataset = dataset

    @property
    def desired_mode(self):
        return self._desired_mode

    @desired_mode.setter
    def desired_mode(self, desired_mode: int):
        """Sets the desired dynamics GP (and builds its posterior)"""
        self.dynamics.desired_mode = desired_mode
        self._desired_mode = desired_mode

    def get_config(self):
        return {
            "start_state": self.start_state,
            "target_state": self.target_state,
            "env_name": self.env_name,
            # "dynamics": tf.keras.layers.serialize(self.dynamics),
            "dynamics": tf.keras.utils.serialize_keras_object(self.dynamics),
            "dataset": (self.dataset[0].numpy(), self.dataset[1].numpy()),
            "desired_mode": self.desired_mode,
            "mode_satisfaction_probability": self.mode_satisfaction_probability,
            # "batch_size": self.batch_size,
            # "num_epochs": self.num_epochs,
            # "validation_split": self.validation_split,
            "learning_rate": self.learning_rate,
            "epsilon": self.epsilon,
            "save_freq": self.save_freq,
            "log_dir": self.log_dir,
            "dynamics_fit_kwargs": self.dynamics_fit_kwargs,
        }

    @classmethod
    def from_config(cls, cfg: dict):
        dynamics = tf.keras.layers.deserialize(
            cfg["dynamics"], custom_objects={"ModeOptDynamics": ModeOptDynamics}
        )
        try:
            log_dir = cfg["log_dir"]
        except KeyError:
            log_dir = "./"
        try:
            dataset = cfg["dataset"]
            dataset = (np.array(dataset[0]), np.array(dataset[1]))
        except (KeyError, TypeError):
            dataset = None
        try:
            dynamics_fit_kwargs = cfg["dynamics_fit_kwargs"]
        except KeyError:
            dynamics_fit_kwargs = DEFAULT_DYNAMICS_FIT_KWARGS
        return cls(
            start_state=try_array_except_none(cfg, "start_state"),
            target_state=try_array_except_none(cfg, "target_state"),
            env_name=cfg["env_name"],
            dynamics=dynamics,
            mode_controller=None,  # TODO set this properly
            dataset=dataset,
            desired_mode=try_val_except_none(cfg, "desired_mode"),
            mode_satisfaction_probability=try_val_except_none(
                cfg, "mode_satisfaction_probability"
            ),
            # batch_size=try_val_except_none(cfg, "batch_size"),
            # num_epochs=try_val_except_none(cfg, "num_epochs"),
            # validation_split=try_val_except_none(cfg, "validation_split"),
            learning_rate=try_val_except_none(cfg, "learning_rate"),
            epsilon=try_val_except_none(cfg, "epsilon"),
            save_freq=try_val_except_none(cfg, "save_freq"),
            log_dir=log_dir,
            dynamics_fit_kwargs=dynamics_fit_kwargs,
        )
