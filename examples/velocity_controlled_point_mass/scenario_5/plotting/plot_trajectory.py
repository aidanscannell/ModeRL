#!/usr/bin/env python3
from modeopt.monitor import ModeOptPlotter, create_test_inputs
from mogpe.helpers.quadcopter_plotter import QuadcopterPlotter
from velocity_controlled_point_mass.utils import init_mode_traj_opt_from_ckpt

if __name__ == "__main__":
    save_dir = "./velocity_controlled_point_mass/scenario_5/images/mode-remaining/mode-conditioning"
    save_dir = "./velocity_controlled_point_mass/scenario_5/images/mode-remaining/riemannian-energy"

    ckpt_dir = "./velocity_controlled_point_mass/scenario_5/logs/learn_dynamics/subset_dataset/white=True/2_experts/batch_size_32/learning_rate_0.01/further_gating_bound/num_inducing_90/12-07-173939/mode_remaining_conditioning_energy_traj_opt/12-08-142854"
    save_dir = "./velocity_controlled_point_mass/scenario_5/images/mode-remaining/conditioning-energy"

    # mode_opt_init = init_mode_opt_from_ckpt(ckpt_dir)
    mode_opt, controls_init, training_spec = init_mode_traj_opt_from_ckpt(ckpt_dir)

    # Create plotter
    test_inputs = create_test_inputs(*mode_opt.dataset, num_test=100)
    # test_inputs = create_test_inputs(*mode_opt.dataset, num_test=10)
    mogpe_plotter = QuadcopterPlotter(
        model=mode_opt.dynamics.mosvgpe,
        X=mode_opt.dataset[0],
        Y=mode_opt.dataset[1],
        test_inputs=test_inputs,
    )

    mode_opt_plotter = ModeOptPlotter(mode_opt=mode_opt, mogpe_plotter=mogpe_plotter)

    mode_opt_plotter.plot_model(save_dir=save_dir)
    # plt.show()
