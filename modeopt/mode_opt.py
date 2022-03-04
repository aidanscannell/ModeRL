#!/usr/bin/env python3
import os
from typing import Callable, List, Optional, Union

import numpy as np
import simenvs
import tensor_annotations.tensorflow as ttf
import tensorflow as tf
from gpflow import default_float
from gpflow.utilities.keras import try_array_except_none, try_val_except_none
from mogpe.keras.utils import save_json_config
from tensor_annotations.axes import Batch

from modeopt.controllers import (
    CONTROLLER_OBJECTS,
    FeedbackController,
    NonFeedbackController,
)
from modeopt.custom_types import Dataset, StateDim
from modeopt.dynamics import ModeOptDynamics
from modeopt.rollouts import rollout_controller_in_dynamics, rollout_controller_in_env

Callback = Callable
Controller = Union[FeedbackController, NonFeedbackController]
JSON_CONFIG_FILENAME = "config.json"

DEFAULT_DYNAMICS_FIT_KWARGS = {
    "batch_size": 16,
    "epochs": 1000,
    "verbose": True,
    "validation_split": 0.2,
}


class ModeOpt(tf.Module):
    def __init__(
        self,
        start_state: ttf.Tensor2[Batch, StateDim],
        target_state: ttf.Tensor2[Batch, StateDim],
        env_name: str,
        dynamics: ModeOptDynamics,
        mode_controller: Controller,
        explorative_controller: Controller = None,
        dataset: Dataset = None,
        desired_mode: int = 1,
        mode_satisfaction_probability: float = 0.7,  # Mode satisfaction probability (0, 1]
        learning_rate: float = 0.01,
        epsilon: float = 1e-8,
        save_freq: Optional[Union[str, int]] = None,
        log_dir: str = "./",
        dynamics_fit_kwargs: dict = DEFAULT_DYNAMICS_FIT_KWARGS,
        max_to_keep: int = 5,
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
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.save_freq = save_freq
        self.log_dir = log_dir
        self.dynamics_fit_kwargs = dynamics_fit_kwargs

        self.mode_controller = mode_controller
        self.mode_controller_callback = []

        self.explorative_controller = explorative_controller

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
            self.dynamics_callbacks.append(
                tf.keras.callbacks.ModelCheckpoint(
                    filepath=os.path.join(log_dir + "ckpts/ModeOptDynamics"),
                    monitor="loss",
                    save_format="tf",
                    save_best_only=True,
                    save_freq=save_freq,
                )
            )

        checkpoint = tf.train.Checkpoint(
            dynamics=self.dynamics,
            mode_controller=self.mode_controller,
            # explorative_controller=self.explorative_controller,
        )
        self.ckpt_dir = os.path.join(log_dir, "ckpts")
        self.ckpt_manager = tf.train.CheckpointManager(
            checkpoint,
            directory=self.ckpt_dir,
            max_to_keep=max_to_keep,
        )

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
        self.save()

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
        self.save()
        self.explorative_controller.optimise()
        self.save()
        return rollout_controller_in_env(
            env=self.env,
            controller=self.explorative_controller,
            start_state=self.start_state,
        )

    def optimise_mode_controller(self):
        self.save()
        self.mode_controller.optimise(self.mode_controller_callback)
        print("SAVING ModeOpt")
        self.save()
        print("SAVED")

    # def check_mode_remaining(self, trajectory):
    #     mode_probs = self.dynamics.predict_mode_probability(state_mean, control_mean)
    #     if (mode_probs < self.mode_satisfaction_probability).any():
    #         return False
    #     else:
    #         return True

    def dynamics_rollout(self):
        return rollout_controller_in_dynamics(
            dynamics=self.dynamics,
            controller=self.mode_controller,
            start_state=self.start_state,
        )

    def env_rollout(self) -> Dataset:
        return rollout_controller_in_env(
            env=self.env, controller=self.mode_controller, start_state=self.start_state
        )

    def add_dynamics_callbacks(self, callbacks: Union[List[Callback], Callback]):
        if isinstance(callbacks, list):
            self.dynamics_callbacks.join(callbacks)
        else:
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

    def save(self):
        """Save checkpoint and json config"""
        self.ckpt_manager.save()
        save_json_config(
            self, filename=os.path.join(self.ckpt_dir, JSON_CONFIG_FILENAME)
        )

    @classmethod
    def load(
        cls, ckpt_dir: str, json_config_filename: Optional[str] = None
    ) -> tf.Module:
        """Load ModeOptimiser from json config and restore variables from checkpoint"""
        if json_config_filename is None:
            json_config_filename = os.path.join(ckpt_dir, JSON_CONFIG_FILENAME)
        with open(json_config_filename, "r") as read_file:
            json_cfg = read_file.read()
        mode_optimiser = tf.keras.models.model_from_json(
            json_cfg, custom_objects={"ModeOpt": ModeOpt}
        )
        ckpt = tf.train.Checkpoint(
            dynamics=mode_optimiser.dynamics,
            mode_controller=mode_optimiser.mode_controller,
            # explorative_controller=mode_optimiser.explorative_controller,
        )
        ckpt.restore(tf.train.latest_checkpoint(ckpt_dir))
        return mode_optimiser

    def get_config(self):
        if self.dataset is not None:
            dataset = (self.dataset[0].numpy(), self.dataset[1].numpy())
        else:
            dataset = None
        return {
            "start_state": self.start_state.numpy(),
            "target_state": self.target_state.numpy(),
            "env_name": self.env_name,
            "dynamics": tf.keras.utils.serialize_keras_object(self.dynamics),
            "mode_controller": tf.keras.utils.serialize_keras_object(
                self.mode_controller
            ),
            # "explorative_controller": tf.keras.utils.serialize_keras_object(
            #     self.explorative_controller
            # ),
            "dataset": dataset,
            "desired_mode": self.desired_mode,
            "mode_satisfaction_probability": self.mode_satisfaction_probability,
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
        mode_controller = tf.keras.layers.deserialize(
            cfg["mode_controller"], custom_objects=CONTROLLER_OBJECTS
        )
        # explorative_controller = tf.keras.layers.deserialize(
        #     cfg["explorative_controller"], custom_objects=CONTROLLER_OBJECTS
        # )
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
        mode_optimiser = cls(
            start_state=tf.constant(
                try_array_except_none(cfg, "start_state"), dtype=default_float()
            ),
            target_state=tf.constant(
                try_array_except_none(cfg, "target_state"), dtype=default_float()
            ),
            env_name=cfg["env_name"],
            dynamics=dynamics,
            mode_controller=mode_controller,
            # mode_controller=None,
            # explorative_controller=explorative_controller,
            dataset=dataset,
            desired_mode=try_val_except_none(cfg, "desired_mode"),
            mode_satisfaction_probability=try_val_except_none(
                cfg, "mode_satisfaction_probability"
            ),
            learning_rate=try_val_except_none(cfg, "learning_rate"),
            epsilon=try_val_except_none(cfg, "epsilon"),
            save_freq=try_val_except_none(cfg, "save_freq"),
            log_dir=log_dir,
            dynamics_fit_kwargs=dynamics_fit_kwargs,
        )
        mode_optimiser.dynamics = dynamics
        return mode_optimiser
