#!/usr/bin/env python3
from typing import Callable, List, Optional, Union

import numpy as np
import tensor_annotations.tensorflow as ttf
import tensorflow as tf
import tensorflow_probability as tfp
from gpflow.models import SVGP
from gpflow.quadrature import NDiagGHQuadrature
from moderl.custom_types import ControlDim, Dataset, Horizon, StateDim
from moderl.utils import combine_state_controls_to_input, save_json_config
from mosvgpe.mixture_of_experts import MixtureOfSVGPExperts
from tensor_annotations.axes import Batch

from .svgp import SVGPDynamicsWrapper

tfd = tfp.distributions

DEFAULT_NUM_GAUSS_HERMITE_POINTS = 20  # Uses too much memory!
DEFAULT_NUM_GAUSS_HERMITE_POINTS = 4

DEFAULT_DYNAMICS_FIT_KWARGS = {
    "batch_size": 16,
    "epochs": 1000,
    "verbose": True,
    "validation_split": 0.2,
}


class DynamicsInterface(tf.keras.Model):
    def call(
        self,
        state_control,
        training: Optional[bool] = False,
        predict_state_difference: Optional[bool] = False,
    ):
        raise NotImplementedError

    def forward(
        self,
        state: Union[tfd.Normal, tfd.Deterministic],
        control: Union[tfd.Normal, tfd.Deterministic],
        predict_state_difference: bool = False,
    ) -> tfd.Distribution:
        raise NotImplementedError

    def optimise(self):
        raise NotImplementedError

    def update_dataset(self, dataset: Dataset):
        raise NotImplementedError


# class ModeRLDynamics(tf.keras.Model):
class ModeRLDynamics(DynamicsInterface):
    def __init__(
        self,
        mosvgpe: MixtureOfSVGPExperts,
        state_dim: int,
        desired_mode: int = 1,
        dataset: Dataset = None,
        learning_rate: float = 0.01,
        epsilon: float = 1e-8,
        dynamics_fit_kwargs: dict = DEFAULT_DYNAMICS_FIT_KWARGS,
        callbacks: Optional[List[Callable]] = None,
        name: str = "ModeRLDynamics",
    ):
        super().__init__(name=name)
        self.mosvgpe = mosvgpe
        self.state_dim = state_dim
        # TODO changed this to get hydra working
        # self._desired_mode = desired_mode
        self.desired_mode = desired_mode
        self.dataset = dataset
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.dynamics_fit_kwargs = dynamics_fit_kwargs
        self._callbacks = callbacks

        # Config the optimiser and compile mosvgpe using keras
        optimiser = tf.keras.optimizers.Adam(
            learning_rate=learning_rate, epsilon=epsilon
        )
        self.mosvgpe.compile(optimizer=optimiser)

    def call(
        self,
        state_control,
        training: Optional[bool] = False,
        predict_state_difference: Optional[bool] = False,
    ):
        """Call the desired mode's GP dynamics model"""
        if not training:
            # TODO update this
            return self.desired_mode_dynamics_gp(
                state=tfd.Deterministic(loc=state_control[:, : self.state_dim]),
                control=tfd.Deterministic(loc=state_control[:, self.state_dim :]),
                predict_state_difference=predict_state_difference,
                add_noise=False,
            )

    def forward(
        self,
        state: Union[tfd.Normal, tfd.Deterministic],  # [horizon, control_dim]
        control: Union[tfd.Normal, tfd.Deterministic],  # [horizon, control_dim]
        predict_state_difference: Optional[bool] = False,
    ) -> tfd.Normal:
        return self.desired_mode_dynamics_gp(
            state=state,
            control=control,
            predict_state_difference=predict_state_difference,
            add_noise=False,
        )

    def optimise(self):
        X, Y = self.dataset
        self.mosvgpe(X)  # Needs to be called to build shapes
        self.mosvgpe.num_data = X.shape[0]  # Needs to be called to build shapes
        # TODO: if callbacks in self.dynamics_fit_kwargs extract and append them
        self.mosvgpe.fit(X, Y, callbacks=self.callbacks, **self.dynamics_fit_kwargs)

    def update_dataset(self, dataset: Dataset):
        if self.dataset is not None:
            Xold, Yold = self.dataset
            Xnew, Ynew = dataset
            X = np.concatenate([Xold, Xnew], 0)
            Y = np.concatenate([Yold, Ynew], 0)
            self.dataset = (X, Y)
        else:
            self.dataset = dataset
        self.mosvgpe.num_data = self.dataset[0].shape[0]

    def predict_mode_probability(
        self,
        state: Union[tfd.Normal, tfd.Deterministic],  # [horizon, control_dim]
        control: Union[tfd.Normal, tfd.Deterministic],  # [horizon, control_dim]
    ) -> ttf.Tensor1[Horizon]:
        h_mean, h_var = self.uncertain_predict_gating(state=state, control=control)
        input_dists = combine_state_controls_to_input(state=state, control=control)
        probs, _ = self.desired_mode_gating_gp.likelihood.predict_mean_and_var(
            input_dists.mean(), h_mean, h_var
        )  # [N, K]
        return probs[:, self.desired_mode]
        # probs = self.mosvgpe.gating_network.predict_mixing_probs_given_h(h_mean, h_var)
        # if probs.shape[-1] == 1:
        #     return probs
        # else:
        #     return probs[:, self.desired_mode]

    def uncertain_predict_gating(
        self,
        state: Union[tfd.Normal, tfd.Deterministic],
        control: Union[tfd.Normal, tfd.Deterministic],
    ):
        # TODO make this handle softmax likelihood (k>2). Just need to map over gps
        input_dists = combine_state_controls_to_input(state=state, control=control)
        # input_mean, input_var = combine_state_controls_to_input(
        #     state_mean=state_mean,
        #     control_mean=control_mean,
        #     state_var=state_var,
        #     control_var=control_var,
        # )
        # if isinstance(input_dists, tfd.Deterministic):
        #     h_mean, h_var = self.desired_mode_gating_gp.predict_f(
        #         input_dists.mean(), full_cov=False
        #     )
        # elif isinstance(input_dists, tfd.Normal):
        #     h_mean, h_var = uncertain_conditional(
        #         input_dists.mean(),
        #         input_dists.variance(),
        #         # input_mean,
        #         # input_var,
        #         self.desired_mode_gating_gp.inducing_variable,
        #         kernel=self.desired_mode_gating_gp.kernel,
        #         q_mu=self.desired_mode_gating_gp.q_mu,
        #         q_sqrt=self.desired_mode_gating_gp.q_sqrt,
        #         mean_function=self.desired_mode_gating_gp.mean_function,
        #         full_output_cov=False,
        #         full_cov=False,
        #         white=self.desired_mode_gating_gp.whiten,
        #     )
        # else:
        #     raise NotImplementedError("input_dists should be Normal or Deterministic")
        # TODO uncomment uncertainty propagation with uncertan_conditional
        h_mean, h_var = self.desired_mode_gating_gp.predict_f(
            input_dists.mean(), full_cov=False
        )
        if self.mosvgpe.gating_network.num_gating_gps == 1:
            h_mean = tf.concat([h_mean, -h_mean], -1)
            h_var = tf.concat([h_var, h_var], -1)
        return h_mean, h_var

    def mode_variational_expectation(
        self,
        state_mean: ttf.Tensor2[Batch, StateDim],
        control_mean: ttf.Tensor2[Batch, ControlDim],
        state_var: ttf.Tensor2[Batch, StateDim] = None,
        control_var: ttf.Tensor2[Batch, ControlDim] = None,
    ):
        """Calculate expected log mode probability under trajectory distribution given by,

        \sum_{t=1}^T \E_{p(\state_t, \control_t)} [\log \Pr(\alpha=k_* \mid \state_t, \control_t)]

        \sum_{t=1}^T \E_{p(\state_t, \control_t, h)} [\log \Pr(\alpha=k_* \mid h(\state_t, \control_t) )]
        """

        input_mean, input_var = combine_state_controls_to_input(
            state_mean, control_mean, state_var, control_var
        )

        def f(Xnew):
            print("Xnew.shape")
            print(Xnew.shape)
            # TODO this only works for Bernoulli likelihood
            # gating_means, gating_vars = self.gating_gp.predict_fs(Xnew, full_cov=False)
            gating_means, gating_vars = self.mosvgpe.gating_network.gp.predict_f(
                Xnew, full_cov=False
            )
            print("gating_means.shape")
            print(gating_means.shape)
            # TODO how to set Y shape if more than two modes?
            Y = tf.ones(gating_means.shape, dtype=default_float()) * (
                self.desired_mode + 1
            )
            # Y = tf.ones(gating_means.shape, dtype=default_float()) * 2
            # Y = tf.ones(gating_means.shape, dtype=default_float())
            # Y = tf.zeros(gating_means.shape, dtype=default_float())
            gating_var_exp = self.mosvgpe.gating_network.gp.likelihood.predict_log_density(
                gating_means,
                gating_vars,
                Y,
                # gating_means[..., self.desired_mode],
                # gating_vars[..., self.desired_mode],
                # Y[..., self.desired_mode],
            )
            return tf.expand_dims(gating_var_exp, -1)

        gauss_quadrature = NDiagGHQuadrature(
            dim=input_mean.shape[-1], n_gh=DEFAULT_NUM_GAUSS_HERMITE_POINTS
        )
        var_exp = gauss_quadrature(f, input_mean, input_var)
        mode_var_exp = tf.reduce_sum(var_exp)

        print("input_mean.shape")
        print(input_mean.shape)
        print(input_var.shape)
        print("var_exp: {}".format(var_exp))
        print("mode_var_exp: {}".format(mode_var_exp))
        return mode_var_exp

    @property
    def desired_mode_dynamics_gp(self) -> SVGPDynamicsWrapper:
        return self._desired_mode_dynamics_gp

    @desired_mode_dynamics_gp.setter
    def desired_mode_dynamics_gp(self, dynamics_gp: SVGP) -> SVGPDynamicsWrapper:
        self._desired_mode_dynamics_gp = SVGPDynamicsWrapper(dynamics_gp)
        # TODO put his print statement back
        # gpf.utilities.print_summary(self._desired_mode_dynamics_gp.svgp_posterior)

    @property
    def desired_mode(self):
        return self._desired_mode

    @desired_mode.setter
    def desired_mode(self, desired_mode: int):
        """Set the desired dynamics mode GP (and build GP posterior)"""
        print("setting desired mode to {}".format(desired_mode))
        assert desired_mode < self.mosvgpe.num_experts
        self._desired_mode = desired_mode
        self.desired_mode_dynamics_gp = self.mosvgpe.experts_list[desired_mode].gp
        self.desired_mode_gating_gp = self.mosvgpe.gating_network.gp

    @property
    def desired_mode_gating_gp(self):
        return self._desired_mode_gating_gp

    @desired_mode_gating_gp.setter
    def desired_mode_gating_gp(self, gp: SVGP):
        # TODO set this differently when K>2
        if self.mosvgpe.gating_network.num_gating_gps == 1:
            self._desired_mode_gating_gp = gp
        else:
            # TODO build a single output gp from a multi output gp
            raise NotImplementedError("How to convert multi output gp to single dim")

    @property
    def gating_gp(self):
        return self.mosvgpe.gating_network.gp

    @property
    def callbacks(self):
        return self._callbacks

    @callbacks.setter
    def callbacks(self, callbacks):
        if self._callbacks is None:
            self._callbacks = callbacks
        else:
            self._callbacks.append(callbacks)

    # @gating_gp.setter
    # def gating_gp(self, gp: SVGP):
    #     # TODO set this differently when K>2
    #     if self.mosvgpe.gating_network.num_gating_gps == 1:
    #         self._gating_gp = gp
    #     else:
    #         # TODO build a single output gp from a multi output gp
    #         raise NotImplementedError("How to convert multi output gp to single dim")

    def save(self, save_filename: str):
        save_json_config(self, filename=save_filename)

    @classmethod
    def load(cls, load_filename: str):
        with open(load_filename, "r") as read_file:
            json_cfg = read_file.read()
        return tf.keras.models.model_from_json(
            json_cfg, custom_objects={"ModeRLDynamics": ModeRLDynamics}
        )

    def get_config(self):
        if self.dataset is not None:
            if isinstance(self.dataset[0], tf.Tensor):
                dataset = (self.dataset[0].numpy(), self.dataset[1].numpy())
            else:
                dataset = (self.dataset[0], self.dataset[1])
        else:
            dataset = None
        return {
            "mosvgpe": tf.keras.layers.serialize(self.mosvgpe),
            "state_dim": self.state_dim,
            "desired_mode": self.desired_mode,
            "learning_rate": self.learning_rate,
            "epsilon": self.epsilon,
            "dynamics_fit_kwargs": self.dynamics_fit_kwargs,
        }

    @classmethod
    def from_config(cls, cfg: dict):
        mosvgpe = tf.keras.layers.deserialize(
            cfg["mosvgpe"],
            custom_objects={"MixtureOfSVGPExperts": MixtureOfSVGPExperts},
        )
        try:
            desired_mode = cfg["desired_mode"]
        except KeyError:
            desired_mode = 1
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
            mosvgpe=mosvgpe,
            state_dim=cfg["state_dim"],
            desired_mode=desired_mode,
            learning_rate=cfg["learning_rate"],
            dataset=dataset,
            epsilon=cfg["epsilon"],
            dynamics_fit_kwargs=dynamics_fit_kwargs,
        )
