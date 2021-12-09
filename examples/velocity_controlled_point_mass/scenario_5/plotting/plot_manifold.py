#!/usr/bin/env python3
import os

import matplotlib.pyplot as plt
import tensorflow as tf
from geoflow.manifolds import GPManifold
from geoflow.plotting import ManifoldPlotter
from gpflow import default_float
from modeopt.monitor import create_test_inputs
from velocity_controlled_point_mass.utils import init_mode_opt_learn_dynamics_from_ckpt

if __name__ == "__main__":
    ckpt_dir = "./velocity_controlled_point_mass/scenario_5/logs/learn_dynamics/subset_dataset/white=True/2_experts/batch_size_32/learning_rate_0.01/further_gating_bound/num_inducing_90/12-07-113403"
    save_dir = "./velocity_controlled_point_mass/scenario_5/images/man"

    (mode_opt, training_spec, train_dataset) = init_mode_opt_learn_dynamics_from_ckpt(
        ckpt_dir
    )

    # manifold = GPManifold(gp=mode_opt.dynamics.gating_gp, covariance_weight=0.01)
    # manifold = GPManifold(gp=mode_opt.dynamics.gating_gp, covariance_weight=1.0)
    # manifold = GPManifold(gp=mode_opt.dynamics.gating_gp, covariance_weight=1000.0)
    # manifold = GPManifold(gp=mode_opt.dynamics.gating_gp, covariance_weight=100.0)
    # manifold = GPManifold(gp=mode_opt.dynamics.gating_gp, covariance_weight=50.0)
    # manifold = GPManifold(gp=mode_opt.dynamics.dynamics_gp, covariance_weight=10.0)
    manifold = GPManifold(gp=mode_opt.dynamics.gating_gp, covariance_weight=10.0)
    # manifold = GPManifold(gp=mode_opt.dynamics.gating_gp, covariance_weight=5.0)
    # manifold = GPManifold(gp=mode_opt.dynamics.gating_gp, covariance_weight=0.05)

    # Create plotter
    test_inputs = create_test_inputs(*mode_opt.dataset, num_test=100)
    plotter = ManifoldPlotter(
        manifold=manifold, test_inputs=tf.constant(test_inputs, dtype=default_float())
    )
    plotter.plot_metric_trace()
    if not os.path.exists(save_dir + "/manifold"):
        os.makedirs(save_dir + "/manifold")
    plt.savefig(save_dir + "/manifold/metric_trace.pdf", transparent=True)
    plotter.plot_jacobian_mean()
    plt.savefig(save_dir + "/manifold/jacobian_mean.pdf", transparent=True)
    plotter.plot_jacobian_var()
    plt.savefig(save_dir + "/manifold/jacobian_cov.pdf", transparent=True)
    # # plt.show()
