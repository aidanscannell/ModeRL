#!/usr/bin/env python3
import os
import shutil

import gin
import gpflow as gpf
import matplotlib.pyplot as plt
import tensorflow as tf
from gpflow import default_float
from matplotlib import patches
from modeopt.monitor import ModeOptPlotter, create_test_inputs
from mogpe.helpers.quadcopter_plotter import QuadcopterPlotter
from scenario_4.utils import config_traj_opt

if __name__ == "__main__":
    # ckpt_dir = "./logs/quadcopter/subset-10/2_experts/batch_size_64/learning_rate_0.01/further_gating_bound/num_inducing_100/11-05-104542"

    # ckpt_dir = "./scenario_4/logs/learn_dynamics/subset_2_dataset/2_experts/batch_size_32/learning_rate_0.01/further_gating_bound/num_inducing_90/11-12-164827/mode_remaining_mode_conditioning_traj_opt/11-16-161704"
    # save_dir = "./scenario_4/images/mode-remaining/mode-conditioning"

    # ckpt_dir = "./scenario_4/logs/learn_dynamics/subset_2_dataset/2_experts/batch_size_32/learning_rate_0.01/further_gating_bound/num_inducing_90/11-10-110119/mode_remaining_riemannian_energy_traj_opt/11-11-093721"
    # save_dir = "./scenario_4/images/mode-remaining/riemannian-energy"

    # ckpt_dir = "./scenario_4/logs/learn_dynamics/subset_2_dataset/2_experts/batch_size_32/learning_rate_0.01/further_gating_bound/num_inducing_90/11-12-164827/mode_remaining_conditioning_energy_traj_opt/11-16-145958"
    ckpt_dir = "./scenario_4/logs/learn_dynamics/subset_2_dataset/2_experts/batch_size_32/learning_rate_0.01/further_gating_bound/num_inducing_90/11-12-164827/mode_remaining_conditioning_energy_traj_opt/11-18-095611"
    save_dir = "./scenario_4/images/mode-remaining/conditioning-energy"

    # ckpt_dir = "./scenario_4/logs/learn_dynamics/subset_3_dataset/2_experts/batch_size_32/learning_rate_0.01/further_gating_bound/num_inducing_90/11-24-162843/mode_remaining_riemannian_energy_traj_opt/11-24-171614"
    # save_dir = "./scenario_4/images/mode-remaining/conditioning-energy/subset_3"

    # ckpt_dir = "./scenario_4/logs/learn_dynamics/subset_2_dataset/2_experts/batch_size_32/learning_rate_0.01/further_gating_bound/num_inducing_90/11-12-164827/baseline_traj_opt/11-16-152246"
    ckpt_dir = "./scenario_4/logs/learn_dynamics/subset_2_dataset/2_experts/batch_size_32/learning_rate_0.01/further_gating_bound/num_inducing_90/11-12-164827/baseline_traj_opt/11-26-093116"
    save_dir = "./scenario_4/images/baseline-traj-opt"

    ckpt_dir = "./scenario_4/logs/learn_dynamics/subset_2_dataset/no_nominal_mean_function/2_experts/batch_size_32/learning_rate_0.01/tight_bound/num_inducing_90/11-29-161235"
    ckpt_dir = "./scenario_4/logs/learn_dynamics/subset_2_dataset/no_nominal_mean_function/2_experts/batch_size_32/learning_rate_0.01/further_gating_bound/num_inducing_90/12-02-155340"
    save_dir = "./scenario_4/images/no-nominal-mean-function"

    ckpt_dir = "./scenario_4/logs/learn_dynamics/svgp_one_mode/2_experts/batch_size_32/learning_rate_0.01/further_gating_bound/num_inducing_90/11-29-160050"
    save_dir = "./scenario_4/images/svgp-no-nominal-mean-function"

    # ckpt_dir = "./scenario_4/logs/learn_dynamics/subset_2_dataset/2_experts/batch_size_32/learning_rate_0.01/further_gating_bound/num_inducing_90/11-10-110119/mode_remaining_chance_constraints_traj_opt/11-10-124104"
    # save_dir = "./scenario_4/images/mode-remaining/chance-constraints"

    ckpt_dir = "./scenario_4/logs/learn_dynamics/subset_2_dataset/basic_kernel_params/white=True/2_experts/batch_size_32/learning_rate_0.01/further_gating_bound/num_inducing_90/12-06-112442/mode_remaining_conditioning_energy_uncertain_traj_opt/12-06-143733"
    # ckpt_dir = "./scenario_4/logs/learn_dynamics/subset_2_dataset/no_nominal_mean_function/white=False/2_experts/batch_size_32/learning_rate_0.01/further_gating_bound/num_inducing_90/12-03-172618/mode_remaining_conditioning_energy_traj_opt/no_nominal_dynamics/12-03-175948"
    # ckpt_dir = "./scenario_4/logs/learn_dynamics/subset_3_dataset/2_experts/batch_size_32/learning_rate_0.01/further_gating_bound/num_inducing_90/11-24-162843/mode_remaining_riemannian_energy_traj_opt/11-24-171614"
    save_dir = "./scenario_4/images/man"

    def init_mode_opt_from_ckpt(ckpt_dir):
        mode_opt_config_file = os.path.join(ckpt_dir, "mode_opt_config.gin")
        # mode_opt_init_config_file = os.path.join(ckpt_dir, "mode_opt_init_config.gin")
        # shutil.copyfile(mode_opt_config_file, mode_opt_init_config_file)
        gin.parse_config_files_and_bindings([mode_opt_config_file], None)
        mode_optimiser, training_spec = config_traj_opt(
            mode_opt_config_file=mode_opt_config_file, log_dir=None
        )
        gpf.utilities.print_summary(mode_optimiser)
        controls_init = mode_optimiser.policy()

        ckpt = tf.train.Checkpoint(model=mode_optimiser)
        manager = tf.train.CheckpointManager(ckpt, ckpt_dir, max_to_keep=5)
        ckpt.restore(manager.latest_checkpoint)
        print("Restored ModeOpt")
        gpf.utilities.print_summary(mode_optimiser)
        return mode_optimiser, controls_init, training_spec

    # mode_opt_init = init_mode_opt_from_ckpt(ckpt_dir)
    mode_opt, controls_init, training_spec = init_mode_opt_from_ckpt(ckpt_dir)

    # Create plotter
    test_inputs = create_test_inputs(*mode_opt.dataset, num_test=100)
    # test_inputs = create_test_inputs(*mode_opt.dataset, num_test=10)
    # def create_test_inputs(X, Y, num_test=400, factor=1.2, const=0.0):
    mogpe_plotter = QuadcopterPlotter(
        model=mode_opt.dynamics.mosvgpe,
        X=mode_opt.dataset[0],
        Y=mode_opt.dataset[1],
        test_inputs=test_inputs,
    )

    # Z = mode_opt.dynamics.dynamics_gp.inducing_variable.inducing_variables[0].Z
    # q_mu = mode_opt.dynamics.dynamics_gp.q_mu
    # q_sqrt = mode_opt.dynamics.dynamics_gp.q_sqrt

    # def plot_inducing_variables(fig, axs, Z, q_mu, q_sqrt, color="b"):
    #     # print("q_mu.shape")
    #     # print(q_mu.shape)
    #     # print(q_sqrt.shape)
    #     q_diag = tf.transpose(tf.linalg.diag_part(q_sqrt))
    #     print("q_diag.shape")
    #     print(q_diag)
    #     # print("plotting inducing_variables")
    #     # tf.print("plotting inducing_variables")
    #     # for ax in axs.flat:
    #     ax = axs[0]
    #     for Z_, q_mu_, q_diag_ in zip(Z.numpy(), q_mu.numpy(), q_diag.numpy()):
    #         ax.add_patch(
    #             patches.Ellipse(
    #                 (Z_[0], Z_[1]),
    #                 # q_diag_[0] * 1.0,
    #                 # q_diag_[1] * 1.0,
    #                 # q_diag_[0] * 0.000001,
    #                 # q_diag_[1] * 0.000001,
    #                 # q_diag_[0] * 1000000000,
    #                 # q_diag_[1] * 1000000000,
    #                 q_diag_[0] * 1,
    #                 q_diag_[0] * 1,
    #                 facecolor="none",
    #                 edgecolor=color,
    #                 linewidth=0.2,
    #                 alpha=1.0,
    #                 # linewidth=0.1,
    #                 # alpha=0.6,
    #             )
    #         )

    # def unwhiten(expert):
    #     Z = expert.inducing_variable.inducing_variables[0].Z
    #     q_sqrt = expert.q_sqrt
    #     print("Z.shape")
    #     print(Z.shape)
    #     Kuu = expert.kernel.K(Z, Z, full_output_cov=False)
    #     print("Kuu")
    #     print(Kuu.shape)
    #     Lu = tf.linalg.cholesky(Kuu)
    #     print("Lu")
    #     print(Lu.shape)
    #     # iLu = tf.linalg.inv(Lu)
    #     # print("iLu")
    #     # print(iLu.shape)
    #     LuT = tf.transpose(Lu, [0, 2, 1])
    #     print(LuT.shape)
    #     S = Lu @ q_sqrt @ LuT
    #     # S = Lu @ q_sqrt
    #     return S

    # colors = ["b", "r"]
    # # fig, axs = mogpe_plotter.plot_experts_f()
    # fig, axs = mogpe_plotter.plot_gating_gps()
    # for ax in axs.flat:
    #     ax.set_xlim([-8.0, 8.0])
    #     ax.set_ylim([-8.0, 8.0])
    # gating_gp = mode_opt.dynamics.gating_gp
    # Z = gating_gp.inducing_variable.Z
    # q_mu = gating_gp.q_mu
    # q_sqrt = gating_gp.q_sqrt
    # # q_sqrt = unwhiten(expert)
    # print("S")
    # print(q_sqrt)
    # plot_inducing_variables(fig, axs[0, :], Z=Z, q_mu=q_mu, q_sqrt=q_sqrt)

    # for k, expert in enumerate(mode_opt.dynamics.mosvgpe.experts.experts_list):
    #     Z = expert.inducing_variable.inducing_variables[0].Z
    #     q_mu = expert.q_mu
    #     q_sqrt = expert.q_sqrt
    #     q_sqrt = unwhiten(expert)
    #     print("S")
    #     print(q_sqrt)
    #     plot_inducing_variables(
    #         fig, axs[k, :], Z=Z, q_mu=q_mu, q_sqrt=q_sqrt, color=colors[k]
    #     )
    # plt.show()

    # mode_opt_plotter = ModeOptPlotter(mode_opt, mogpe_plotter)
    mode_opt_plotter = ModeOptPlotter(
        mode_opt=mode_opt,
        # control_means_init=controls_init[0],
        # control_vars_init=controls_init[1],
        mogpe_plotter=mogpe_plotter,
    )

    from geoflow.manifolds import GPManifold
    from geoflow.plotting import ManifoldPlotter

    # manifold = GPManifold(gp=mode_opt.dynamics.gating_gp, covariance_weight=0.01)
    # manifold = GPManifold(gp=mode_opt.dynamics.gating_gp, covariance_weight=1.0)
    # manifold = GPManifold(gp=mode_opt.dynamics.gating_gp, covariance_weight=1000.0)
    # manifold = GPManifold(gp=mode_opt.dynamics.gating_gp, covariance_weight=100.0)
    # manifold = GPManifold(gp=mode_opt.dynamics.gating_gp, covariance_weight=50.0)
    manifold = GPManifold(gp=mode_opt.dynamics.gating_gp, covariance_weight=5.0)
    # manifold = GPManifold(gp=mode_opt.dynamics.gating_gp, covariance_weight=5.0)
    # manifold = GPManifold(gp=mode_opt.dynamics.gating_gp, covariance_weight=0.05)
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

    mode_opt_plotter.plot_model(save_dir=save_dir)
    # plt.show()
