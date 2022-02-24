#!/usr/bin/env python3
from modeopt.monitor import ModeOptPlotter, create_test_inputs
from mogpe.helpers.quadcopter_plotter import QuadcopterPlotter
from velocity_controlled_point_mass.utils import init_mode_traj_opt_from_ckpt
from modeopt.metrics import (
    approximate_riemannian_energy,
    gating_function_variance,
    mode_probability,
)


def plot_ckpt(ckpt_dir, save_dir):
    # mode_opt_init = init_mode_opt_from_ckpt(ckpt_dir)
    mode_opt, controls_init, training_spec = init_mode_traj_opt_from_ckpt(ckpt_dir)

    # Create plotter
    # test_inputs = create_test_inputs(*mode_opt.dataset, num_test=400)
    test_inputs = create_test_inputs(*mode_opt.dataset, num_test=10000)
    # test_inputs = create_test_inputs(*mode_opt.dataset, num_test=10)
    mogpe_plotter = QuadcopterPlotter(
        model=mode_opt.dynamics.mosvgpe,
        X=mode_opt.dataset[0],
        Y=mode_opt.dataset[1],
        test_inputs=test_inputs,
    )
    print("mogpe_plotter")
    print(type(mogpe_plotter))

    mode_opt_plotter = ModeOptPlotter(
        env=mode_opt.env, mode_opt=mode_opt, mogpe_plotter=mogpe_plotter
    )

    mode_opt_plotter.plot_model(save_dir=save_dir)
    # plt.show()


def calculate_metrics_given_ckpt_dir(ckpt_dir: str, save_dir: str) -> None:
    covariance_weight = 1.0
    sum = True
    mode_opt, controls_init, training_spec = init_mode_traj_opt_from_ckpt(ckpt_dir)

    prob = mode_probability(mode_opt, sum=sum)
    gating_variance = gating_function_variance(mode_opt, sum=sum)
    riemannian_energy = approximate_riemannian_energy(
        mode_opt, covariance_weight=covariance_weight, sum=sum
    )

    print("Desired prob: {}".format(prob))
    print("Gating function variance: {}".format(gating_variance))
    print("Riemannian energy: {}".format(riemannian_energy))


if __name__ == "__main__":

    ckpt_dir = "./velocity_controlled_point_mass/scenario_5/logs/learn_dynamics/subset_dataset/white=True/2_experts/batch_size_32/learning_rate_0.01/further_gating_bound/num_inducing_90/12-07-173939/mode_remaining_conditioning_energy_traj_opt/12-08-142854"
    save_dir = "./velocity_controlled_point_mass/scenario_5/images/mode-remaining/conditioning-energy"

    ckpt_dir = "./velocity_controlled_point_mass/scenario_5/learn_dynamics/subset_dataset/white=True/2_experts/batch_size_32/learning_rate_0.01/further_gating_bound/num_inducing_90/12-10-124629/mode_remaining_prob_cost_fn_traj_opt/12-14-094132"
    save_dir = "./velocity_controlled_point_mass/scenario_5/images/mode-remaining/prob"

    ckpt_dict = {
        # "baseline-svgp-traj-opt": "./velocity_controlled_point_mass/scenario_5/logs/learn_dynamics/single_svgp/2_experts/batch_size_32/learning_rate_0.01/further_gating_bound/num_inducing_90/12-20-122600/svgp_traj_opt/12-20-142749",
        # "baseline-traj-opt": "./velocity_controlled_point_mass/scenario_5/logs/learn_dynamics/subset_dataset/white=True/no-nominal/zero-mean-func/2_experts/batch_size_32/learning_rate_0.01/further_gating_bound/num_inducing_90/12-15-135504/baseline_traj_opt/12-15-150625",
        # "mode-conditioning-max-entropy": "./velocity_controlled_point_mass/scenario_5/logs/learn_dynamics/subset_dataset/white=True/no-nominal/zero-mean-func/2_experts/batch_size_32/learning_rate_0.01/further_gating_bound/num_inducing_90/12-15-135504/mode_remaining_mode_conditioning_traj_opt/12-15-171345",
        # "mode-conditioning-no-max-entropy": "./velocity_controlled_point_mass/scenario_5/logs/learn_dynamics/subset_dataset/white=True/no-nominal/zero-mean-func/2_experts/batch_size_32/learning_rate_0.01/further_gating_bound/num_inducing_90/12-15-135504/mode_remaining_mode_conditioning_traj_opt/12-15-171059",
        # "riemannian-energy-low": "./velocity_controlled_point_mass/scenario_5/logs/learn_dynamics/subset_dataset/white=True/no-nominal/zero-mean-func/2_experts/batch_size_32/learning_rate_0.01/further_gating_bound/num_inducing_90/12-15-135504/mode_remaining_riemannian_energy_traj_opt/12-15-142411",
        # "riemannian-energy": "./velocity_controlled_point_mass/scenario_5/logs/learn_dynamics/subset_dataset/white=True/no-nominal/zero-mean-func/2_experts/batch_size_32/learning_rate_0.01/further_gating_bound/num_inducing_90/12-15-135504/mode_remaining_riemannian_energy_low_cov_traj_opt/12-16-111548",
        # "riemannian-energy-high": "./velocity_controlled_point_mass/scenario_5/logs/learn_dynamics/subset_dataset/white=True/no-nominal/zero-mean-func/2_experts/batch_size_32/learning_rate_0.01/further_gating_bound/num_inducing_90/12-15-135504/mode_remaining_riemannian_energy_high_cov_traj_opt/12-16-112615",
        "riemannian-energy-high5": "./velocity_controlled_point_mass/scenario_5/logs/learn_dynamics/subset_dataset/white=True/no-nominal/zero-mean-func/2_experts/batch_size_32/learning_rate_0.01/further_gating_bound/num_inducing_90/12-15-135504/mode_remaining_riemannian_energy_high_cov_traj_opt/01-07-112204",
        # "conditioning-energy-no-max-entropy": "./velocity_controlled_point_mass/scenario_5/logs/learn_dynamics/subset_dataset/white=True/no-nominal/zero-mean-func/2_experts/batch_size_32/learning_rate_0.01/further_gating_bound/num_inducing_90/12-15-135504/mode_remaining_riemannian_energy_high_cov_traj_opt/12-16-112615",
        "conditioning-energy-high5-max-entropy": "./velocity_controlled_point_mass/scenario_5/logs/learn_dynamics/subset_dataset/white=True/no-nominal/zero-mean-func/2_experts/batch_size_32/learning_rate_0.01/further_gating_bound/num_inducing_90/12-15-135504/mode_remaining_conditioning_energy_max_entropy_traj_opt/01-07-112915",
    }
    # ckpt_dir = "./velocity_controlled_point_mass/scenario_5/learn_dynamics/subset_dataset/white=True/no-nominal/zero-mean-func/2_experts/batch_size_32/learning_rate_0.01/further_gating_bound/num_inducing_90/12-15-135504/mode_remaining_mode_conditioning_traj_opt/12-15-171345"
    # save_dir = "./velocity_controlled_point_mass/scenario_5/images/mode-remaining/mode-conditioning/max-entropy"
    # ckpt_dir = "./velocity_controlled_point_mass/scenario_5/learn_dynamics/subset_dataset/white=True/no-nominal/zero-mean-func/2_experts/batch_size_32/learning_rate_0.01/further_gating_bound/num_inducing_90/12-15-135504/mode_remaining_mode_conditioning_traj_opt/12-15-171059"
    # save_dir = "./velocity_controlled_point_mass/scenario_5/images/mode-remaining/mode-conditioning/no-max-entropy"

    # ckpt_dir = "./velocity_controlled_point_mass/scenario_5/learn_dynamics/subset_dataset/white=True/no-nominal/zero-mean-func/2_experts/batch_size_32/learning_rate_0.01/further_gating_bound/num_inducing_90/12-15-135504/baseline_traj_opt/12-15-150625"
    # save_dir = "./velocity_controlled_point_mass/scenario_5/images/baseline-traj-opt"

    # ckpt_dir = "./velocity_controlled_point_mass/scenario_5/learn_dynamics/subset_dataset/white=True/2_experts/batch_size_32/learning_rate_0.01/further_gating_bound/num_inducing_90/12-10-124629/mode_remaining_prob_var_cost_fn_traj_opt/12-14-102401"
    # save_dir = (
    #     "./velocity_controlled_point_mass/scenario_5/images/mode-remaining/prob-var"
    # )

    for key in ckpt_dict.keys():
        save_dir = "./velocity_controlled_point_mass/scenario_5/images/" + key
        print("save_dir")
        print(save_dir)
        print(ckpt_dict[key])
        plot_ckpt(ckpt_dict[key], save_dir=save_dir)
        calculate_metrics_given_ckpt_dir(ckpt_dict[key], save_dir=save_dir)
