#!/usr/bin/env python3
from modeopt.cost_functions import quadratic_cost_fn
from functools import partial
from typing import Callable, Optional

import gpflow as gpf
import tensor_annotations.tensorflow as ttf
import tensorflow as tf
import tensorflow_probability as tfp
from gpflow import default_float
from modeopt.custom_types import ControlMeanAndVariance, StateDim
from modeopt.dynamics import ModeOptDynamics
from modeopt.trajectories import GeodesicTrajectory, TRAJECTORY_OBJECTS

from ..base import NonFeedbackController
from .trajectory_optimisation import TrajectoryOptimisationController

tfd = tfp.distributions


class GeodesicController(NonFeedbackController):
    def __init__(
        self,
        start_state: ttf.Tensor1[StateDim],
        target_state: ttf.Tensor1[StateDim],
        dynamics: ModeOptDynamics,
        # Collocation args
        horizon: int = 10,
        t_init: float = -1.0,
        t_end: float = 1.0,
        riemannian_metric_covariance_weight: float = 1.0,
        max_collocation_iterations: int = 100,
        collocation_constraints_lower: float = -0.1,
        collocation_constraints_upper: float = 0.1,
        dummy_cost_weight: Optional[ttf.Tensor2[StateDim, StateDim]] = None,
        keep_last_solution: bool = True,
        # Control inference args
        num_inference_iterations: int = 100,  # num optimisation steps to use for control inference
        num_control_samples: int = 1,  # number of samples to draw from control variational posterior
        method: Optional[str] = "SLSQP",
        name: str = "GeodesicController",
    ):
        super().__init__(name=name)
        self.dynamics = dynamics
        self.collocation_constraints_lower = collocation_constraints_lower
        self.collocation_constraints_upper = collocation_constraints_upper
        self.dummy_cost_weight = dummy_cost_weight
        self.num_inference_iterations = num_inference_iterations
        self.num_control_samples = num_control_samples
        self.optimiser = tf.keras.optimizers.Adam(
            learning_rate=0.1
        )  # optimiser for control inference

        initial_solution = GeodesicTrajectory(
            start_state=start_state,
            target_state=target_state,
            gp=dynamics.desired_mode_gating_gp,
            riemannian_metric_covariance_weight=riemannian_metric_covariance_weight,
            horizon=horizon,
            t_init=t_init,
            t_end=t_end,
        )

        if dummy_cost_weight is not None:

            def objective_fn(initial_solution: GeodesicTrajectory):
                """Dummy cost function that regularises the trajectory"""
                costs = quadratic_cost_fn(
                    # vector=initial_solution.states,
                    vector=initial_solution.state_derivatives,
                    weight_matrix=dummy_cost_weight,
                    vector_var=None,
                )
                return tf.reduce_sum(costs)

        else:
            objective_fn = lambda initial_solution: 1.0

        self.trajectory_optimiser = TrajectoryOptimisationController(
            max_collocation_iterations,
            initial_solution,
            objective_fn,
            keep_last_solution=keep_last_solution,
            nonlinear_constraint_closure=initial_solution.geodesic_collocation_constraints,
            nonlinear_constraint_kwargs={
                "lb": collocation_constraints_lower,
                "ub": collocation_constraints_upper,
            },
            method=method,
        )

        # Initialise prior/variational posterior for control inference (from states)
        self.controls_prior = tfd.MultivariateNormalDiag(
            loc=tf.zeros(
                (initial_solution.horizon, initial_solution.state_dim),
                dtype=default_float(),
            ),
            scale_diag=tf.ones(
                (initial_solution.horizon, initial_solution.state_dim),
                dtype=default_float(),
            ),
        )
        self.controls_posterior = tfd.MultivariateNormalDiag(
            loc=tf.Variable(
                tf.zeros(
                    (initial_solution.horizon, initial_solution.state_dim),
                    dtype=default_float(),
                ),
                trainable=False,
            ),
            scale_diag=tf.Variable(
                tf.ones(
                    (initial_solution.horizon, initial_solution.state_dim),
                    dtype=default_float(),
                )
                * 1.0,  # TODO make this use control_dim?
                trainable=False,
            ),
        )
        gpf.utilities.print_summary(self)

    def __call__(
        self, timestep: Optional[int] = None, variance: bool = False
    ) -> ControlMeanAndVariance:
        # return self.previous_solution(timestep=timestep, variance=variance)
        if timestep is not None:
            idxs = [timestep, ...]
        else:
            idxs = [...]
        if variance:
            return self.controls[idxs], None
            # return self.controls[idxs], self.control_vars[idxs]
        else:
            return self.controls[idxs]

    @property
    def controls(self):
        return self.controls_posterior.mean()

    def optimise(
        self, callback: Optional[Callable[[tf.Tensor, tf.Tensor, int], None]] = []
    ):
        # Optimise state trajectory
        optimisation_result = self.trajectory_optimiser.optimise(callback)
        self.infer_controls_from_states(callback, num_steps=optimisation_result.nit)

    def infer_controls_from_states(
        self,
        callback: Optional[Callable[[tf.Tensor, tf.Tensor, int], None]] = [],
        num_steps: int = 0,
    ):
        gpf.utilities.set_trainable(self.controls_posterior, True)

        @tf.function
        def optimisation_step():
            with tf.GradientTape() as tape:
                negative_elbo = -self.latent_variable_elbo()
            gradients = tape.gradient(
                negative_elbo, self.controls_posterior.trainable_variables
            )
            self.optimiser.apply_gradients(
                zip(gradients, self.controls_posterior.trainable_variables)
            )

        for step in range(self.num_inference_iterations):
            optimisation_step()
            callback(step + num_steps, None, None)

        gpf.utilities.set_trainable(self.controls_posterior, False)
        print("Control inference result:")
        print(self.controls)

    def latent_variable_elbo(self):
        # num_samples = 1
        control_samples = self.controls_posterior.sample(self.num_control_samples)
        # control_samples = tf.expand_dims(self.controls_posterior.mean(), 0)
        states_broadcast = tf.broadcast_to(
            tf.expand_dims(self.previous_solution.states, 0),
            shape=(self.num_control_samples, *self.previous_solution.states.shape),
        )
        state_control_inputs = tf.concat([states_broadcast, control_samples], -1)
        state_derivatives = tf.broadcast_to(
            tf.expand_dims(self.previous_solution.state_derivatives, 0),
            shape=(
                self.num_control_samples,
                *self.previous_solution.state_derivatives.shape,
            ),
        )

        data = (state_control_inputs, state_derivatives)
        elbo_closure = partial(
            self.dynamics.mosvgpe.elbo,
            num_samples=1,
            num_data=data[0].shape[0],
            # bound="tight",
            bound="further_gating",
        )
        elbo_samples = tf.map_fn(
            self.dynamics.mosvgpe.experts_list[self.dynamics.desired_mode].gp.elbo,
            # elbo_closure,
            # self.dynamics.mosvgpe.elbo,
            data,
            fn_output_signature=(tf.float64),
        )
        print("elbo")
        print(elbo_samples)
        elbo = tf.reduce_mean(elbo_samples)  # Average samples from control dist

        # KL[q(x) || p(x)]
        KL = tfp.distributions.kl_divergence(
            self.controls_posterior, self.controls_prior
        )
        return elbo - tf.reduce_sum(KL)

    @property
    def initial_solution(self):
        return self.trajectory_optimiser.initial_solution

    @property
    def previous_solution(self):
        return self.trajectory_optimiser.previous_solution

    def get_config(self):
        return {
            "start_state": self.initial_solution.start_state.numpy(),
            "target_state": self.initial_solution.target_state.numpy(),
            "dynamics": tf.keras.utils.serialize_keras_object(self.dynamics),
            "horizon": self.initial_solution.horizon,
            "t_init": self.initial_solution.times[0].numpy(),
            "t_end": self.initial_solution.times[-1].numpy(),
            "riemannian_metric_covariance_weight": self.initial_solution.manifold.covariance_weight,
            "max_collocation_iterations": self.trajectory_optimiser.max_iterations,
            "collocation_constraints_lower": self.collocation_constraints_lower,
            "collocation_constraints_upper": self.collocation_constraints_upper,
            "dummy_cost_weight": self.dummy_cost_weight.numpy(),
            "keep_last_solution": self.trajectory_optimiser.keep_last_solution,
            "num_inference_iterations": self.num_inference_iterations,
            "num_control_samples": self.num_control_samples,
            "method": self.trajectory_optimiser.method,
        }

    @classmethod
    def from_config(cls, cfg: dict):
        dynamics = tf.keras.layers.deserialize(
            cfg["dynamics"], custom_objects={"ModeOptDynamics": ModeOptDynamics}
        )
        controller = cls(
            start_state=tf.constant(cfg["start_state"], dtype=default_float()),
            target_state=tf.constant(cfg["target_state"], dtype=default_float()),
            dynamics=dynamics,
            horizon=cfg["horizon"],
            t_init=cfg["t_init"],
            t_end=cfg["t_end"],
            riemannian_metric_covariance_weight=cfg[
                "riemannian_metric_covariance_weight"
            ],
            max_collocation_iterations=cfg["max_collocation_iterations"],
            collocation_constraints_lower=cfg["collocation_constraints_lower"],
            collocation_constraints_upper=cfg["collocation_constraints_upper"],
            dummy_cost_weight=tf.constant(
                cfg["dummy_cost_weight"], dtype=default_float()
            ),
            keep_last_solution=cfg["keep_last_solution"],
            num_inference_iterations=cfg["num_inference_iterations"],
            num_control_samples=cfg["num_control_samples"],
            method=cfg["method"],
        )
        return controller
