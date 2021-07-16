#!/usr/bin/env python3
import abc
from typing import Callable, Optional

from geoflow.manifolds import GPManifold
import tensorflow as tf
from gpflow.conditionals import uncertain_conditional
from gpflow.config import default_float
from gpflow.likelihoods.utils import inv_probit
from gpflow.models import BayesianModel, GPModel

from vimpc.dynamics import BayesianDynamics, Dynamics, EnvDynamics
from vimpc.policies import VariationalPolicy


def mode_probability(gp, state, control, state_var, control_var):
    input_mean = tf.concat([state, control], -1)
    input_var = tf.concat([state_var, control_var], -1)
    h_mean, h_var = uncertain_conditional(
        input_mean,
        input_var,
        gp.inducing_variable,
        kernel=gp.kernel,
        q_mu=gp.q_mu,
        q_sqrt=gp.q_sqrt,
        mean_function=gp.mean_function,
        full_output_cov=False,
        full_cov=False,
        white=gp.whiten,
    )
    # h_mean, h_var = gp.predict_f(input_mean, full_cov=False)
    # h_mean, h_var = gp.predict_f(state, full_cov=False)
    prob = inv_probit(h_mean / tf.sqrt(1 + h_var))
    # probs = gp.predict_mixing_probs(state, full_cov=False)
    # return probs[:, 1]
    # var = prob - tf.square(prob)
    return 1 - prob


class Controller(abc.ABC):
    def __init__(self, cost_fn: Callable, terminal_cost_fn: Callable):
        self._cost_fn = cost_fn
        self._terminal_cost_fn = terminal_cost_fn

    def cost_fn(self, state, control, state_var=None, control_var=None):
        return self._cost_fn(
            state, control, state_var=state_var, control_var=control_var
        )

    def terminal_cost_fn(self, state, control, state_var=None, control_var=None):
        return self._terminal_cost_fn(
            state,
            control,
            terminal_state_var=state_var,
            terminal_control_var=control_var,
        )


class MPC(Controller):
    def __init__(
        self, cost_fn: Callable, terminal_cost_fn: Callable, dynamics: Dynamics
    ):
        super().__init__(cost_fn=cost_fn, terminal_cost_fn=terminal_cost_fn)
        self._dynamics = dynamics

    @property
    def dynamics(self):
        return self._dynamics


class VIMPC(MPC, BayesianModel):
    def __init__(
        self,
        cost_fn: Callable,
        terminal_cost_fn: Callable,
        dynamics: BayesianDynamics,
        gating_gp: GPModel,
        policy: VariationalPolicy,
        monotonic_fn: Optional[Callable] = None,
    ):
        super().__init__(
            cost_fn=cost_fn, terminal_cost_fn=terminal_cost_fn, dynamics=dynamics
        )
        self._policy = policy
        self.num_time_steps = policy.num_time_steps
        self.gating_gp = gating_gp
        if monotonic_fn is None:
            self._monotoic_fn = tf.exp
        else:
            self._monotoic_fn = monotonic_fn

        # covariance_weight = 0.0
        # covariance_weight = 1.0
        # self.manifold = GPManifold(gp=gating_gp, covariance_weight=covariance_weight)

    def likelihood(self):
        cost = self.cost_fn(state, control)
        return self._monotonic_fn(cost)

    def maximum_log_likelihood_objective(self):
        return -self.elbo

    def mode_probability(self, state, control, state_var=None, control_var=None):
        prob = mode_probability(self.gating_gp, state, control, state_var, control_var)
        return prob

    def training_loss(self, start_state):
        return -self.elbo(start_state)

    def elbo(self, start_state):
        if len(start_state.shape) == 1:
            next_state_mean = tf.reshape(start_state, [1, -1])
        elif len(start_state.shape) == 2:
            assert start_state.shape[0] == 1
            next_state_mean = start_state
        else:
            raise AttributeError("Start state is wrong shape")
        state_dim = start_state.shape[-1]

        # calculate entropy of policy dist
        entropy = self._policy.entropy()

        # get control trajectories variational posterior
        control_means = self._policy.variational_dist.mean()
        control_vars = self._policy.variational_dist.variance()

        # rollout trajectory using dynamics and policy dist
        expected_costs = []
        next_state_var = tf.ones([1, state_dim], dtype=default_float()) * 0.0
        state_means = next_state_mean
        state_vars = next_state_var
        # state_diffs = []
        # state_diff_vars = state_var
        for t in range(self.num_time_steps):
            next_state_mean, next_state_var = self.dynamics(
                next_state_mean,
                control_means[t : t + 1, :],
                state_var=next_state_var,
                control_var=control_vars[t : t + 1, :],
            )
            expected_cost = self.cost_fn(
                next_state_mean,
                control_means[t : t + 1, :],
                state_var=next_state_var,
                control_var=control_vars[t : t + 1, :],
            )
            expected_costs.append(expected_cost)

            state_means = tf.concat([state_means, next_state_mean], 0)
            state_vars = tf.concat([state_vars, next_state_var], 0)
            # state_diff = next_state_mean - state_mean
            # state_diffs.append(state_diff)

        terminal_cost = self.terminal_cost_fn(
            next_state_mean,
            control_means[t : t + 1, :],
            state_var=next_state_var,
            control_var=control_vars[t : t + 1, :],
        )

        expected_costs = tf.reshape(tf.stack(expected_costs, 0), [-1])
        # print("expected_costs")
        # print(expected_costs)

        mode_probs = self.mode_probability(
            state_means[1:, :],
            control_means,
            # state_var=state_vars[1:, :] * 0.0,
            # control_var=control_vars * 0.0,
            state_var=state_vars[1:, :],
            control_var=control_vars,
        )
        log_mode_probs = tf.math.log(mode_probs)
        log_mode_probs_sum = tf.reduce_sum(log_mode_probs)

        # state_diffs = tf.concat(state_diffs, 0)
        # energy = self.manifold.energy(state_means[:-1, :], state_diffs)
        # print("energy")
        # print(energy)

        # metric = self.manifold.metric(state_means)
        # state_means = tf.expand_dims(state_means, 1)
        # metric_inner = state_means @ metric @ tf.transpose(state_means, [0, 2, 1])
        # metric_inner = tf.reduce_sum(metric_inner)

        # calcualte the evidence lower bouned
        elbo = log_mode_probs_sum + entropy - terminal_cost - expected_costs
        return tf.reduce_sum(elbo)


class DeterministicVIMPC(MPC, BayesianModel):
    def __init__(
        self,
        start_state,
        target_state,
        cost_fn: Callable,
        dynamics: EnvDynamics,
        policy: VariationalPolicy,
        monotonic_fn: Optional[Callable] = None,
    ):
        super().__init__(cost_fn=cost_fn, dynamics=dynamics)
        self._policy = policy
        self.start_state = start_state
        self.target_state = target_state
        self.num_time_steps = policy.num_time_steps
        if monotonic_fn is None:
            self._monotoic_fn = tf.exp
        else:
            self._monotoic_fn = monotonic_fn

    def likelihood(self):
        cost = self.cost_fn(state, control)
        return self._monotonic_fn(cost)

    def cost_fn(self, state, control, state_var=None, control_var=None):
        return self._cost_fn(
            state, control, state_var=state_var, control_var=control_var
        )

    def maximum_log_likelihood_objective(self):
        return self.elbo

    def elbo(self, start_state):
        if len(start_state.shape) == 1:
            state = tf.reshape(start_state, [1, -1])
        elif len(start_state.shape) == 2:
            assert start_state.shape[0] == 1
            state = start_state
        else:
            raise AttributeError("Start state is wrong shape")
        state_dim = start_state.shape[-1]

        self.dynamics.reset()

        control_means = self._policy.variational_dist.mean()
        control_vars = self._policy.variational_dist.variance()
        control_dim = control_means.shape[-1]

        expected_costs = []
        state_var = tf.ones(1, state_dim) * 0.000001
        print("state_var")
        print(state_var)
        for t in range(self.num_time_steps):
            next_state = self.dynamics(control_means[t : t + 1, :])
            print("next_state")
            print(next_state.shape)
            state_var = state_var + control_vars[t : t + 1, :]
            print("state_var")
            print(state_var)
            expected_cost = self.cost_fn(
                next_state,
                control_means[t : t + 1, :],
                state_var=state_var,
                control_var=control_vars[t : t + 1, :],
            )
            print("expected_cost")
            print(expected_cost)
            expected_costs.append(expected_cost)
            state = next_state
        # terminal cost
        state_error = next_state - self.target_state
        Q = tf.eye(state_dim, dtype=default_float())
        # terminal_cost = state_error @ Q @ tf.transpose(state_error) + tf.linalg.trace(
        #     state_var @ Q
        # )
        print("terminal_cost")
        print(terminal_cost.shape)

        expected_costs = tf.reshape(tf.stack(expected_costs, 0), [-1])
        print("expected_costs")
        print(expected_costs)
        entropy = self._policy.variational_dist.entropy()
        print("entropy")
        print(entropy)
        # elbo = entropy - expected_costs - terminal_cost
        elbo = entropy - terminal_cost
        print("elbo")
        print(elbo)
        return tf.reduce_sum(elbo)

    # def elbo(self, start_state):
    #     if len(start_state.shape) == 1:
    #         state = tf.reshape(start_state, [1, -1])
    #     elif len(start_state.shape) == 2:
    #         assert start_state.shape[0] == 1
    #         state = start_state
    #     else:
    #         raise AttributeError("Start state is wrong shape")

    #     num_control_samples = 100
    #     num_state_samples = 100
    #     num_control_samples = 1
    #     # num_control_samples = 10
    #     num_state_samples = 3

    #     control_samples_all = self._policy.sample(num_control_samples)
    #     print("control_samples_all")
    #     print(control_samples_all.shape)
    #     # control_traj = self._policy()
    #     control_dim = control_samples_all.shape[-1]
    #     state_dim = start_state.shape[-1]
    #     self.dynamics.reset()
    #     # env.reset(tf.constant(state))

    #     log_prob = self._policy.variational_dist.log_prob(control_samples_all)
    #     # print("log_prob")
    #     # print(log_prob)
    #     sum_log_prob = tf.reduce_sum(log_prob) / num_control_samples

    #     cumulative_cost = 0
    #     cumulative_costs = []
    #     costs = []
    #     # xx, _ = tf.meshgrid(state_samples, control_samples[:, 0, :])
    #     state_broadcast = tf.tile(state, [num_control_samples, 1])
    #     print("state_broadcast")
    #     print(state_broadcast)
    #     for i in range(num_control_samples):
    #         control_traj = control_samples_all[i, :, :]
    #         for t in range(self.num_time_steps):
    #             # control_samples = control_samples_all[:, t, :]
    #             print("control_traj")
    #             print(control_traj)
    #             # next_time_step = self.dynamics.env.step(control_traj[t : t + 1, :])
    #             next_state = self.dynamics(control_traj[t : t + 1, :])
    #             # next_state = next_time_step.observation
    #             print("next_state")
    #             print(next_state.shape)
    #             state = next_state
    #             # next_state_broadcast = tf.expand_dims(next_state, 1)
    #             # control_broadcast = tf.expand_dims(control_samples, 1)
    #             cumulative_costs.append(
    #                 self.cost_fn(next_state, control_traj[t : t + 1, :])
    #             )
    #             print("cumulative_cost")
    #             print(cumulative_cost)
    #             # likelihood += self.likelihood(next_state, control)
    #         print("end state")
    #         print(state)
    #         state_error = next_state - self.target_state
    #         print("state_error")
    #         print(state_error)
    #         # state_error = tf.expand_dims(state_error, 1)
    #         Q = tf.eye(state_dim, dtype=default_float())
    #         terminal_state_cost = state_error @ Q @ tf.transpose(state_error)

    #         print("terminal_state_cost")
    #         print(terminal_state_cost.shape)
    #         cumulative_cost = tf.reduce_sum(cumulative_costs)
    #         costs.append(cumulative_cost + terminal_state_cost)
    #         # costs.append(cumulative_cost)
    #         # costs.append(terminal_state_cost)

    #     cost = tf.reduce_sum(tf.exp(costs)) / num_control_samples
    #     print("cost")
    #     print(cost.shape)
    #     # return -cost
    #     return tf.math.log(-cost) - sum_log_prob
    #     # return -cost - sum_log_prob
