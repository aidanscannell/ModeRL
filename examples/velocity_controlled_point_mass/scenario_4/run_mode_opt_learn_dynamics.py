#!/usr/bin/env python3
import gin
import tensorflow as tf
from gpflow.monitor import Monitor
from gpflow.utilities import print_summary
from modeopt.dynamics import ModeOptDynamicsTrainingSpec
from modeopt.monitor import create_test_inputs
from mogpe.helpers.plotter import Plotter2D
from mogpe.helpers.quadcopter_plotter import QuadcopterPlotter
from mogpe.training.utils import (
    create_log_dir,
    create_tf_dataset,
    init_fast_tasks_bounds,
)

from scenario_4.data.load_data import load_vcpm_dataset
from scenario_4.utils import init_mode_opt, init_checkpoint_manager


@gin.configurable
def run_mode_opt_learn_dynamics(
    mode_opt_config,
    train_dataset,
    test_dataset,
    log_dir,
    num_epochs,
    batch_size,
    learning_rate,
    # optimiser,
    logging_epoch_freq,
    fast_tasks_period,
    slow_tasks_period,
    num_ckpts,
    mogpe_config_file,
    compile_loss_fn: bool = True,
):
    mode_optimiser = init_mode_opt(
        dataset=train_dataset, mogpe_config_file=mogpe_config_file
    )
    print_summary(mode_optimiser)

    # Create monitor tasks (plots/elbo/model params etc)
    print("bound")
    print(mode_optimiser.dynamics.mosvgpe.bound)
    log_dir = create_log_dir(
        log_dir,
        mode_optimiser.dynamics.mosvgpe.num_experts,
        batch_size,
        # learning_rate=optimiser.learning_rate,
        learning_rate=learning_rate,
        bound=mode_optimiser.dynamics.mosvgpe.bound,
        num_inducing=mode_optimiser.dynamics.mosvgpe.experts.experts_list[0]
        .inducing_variable.inducing_variables[0]
        .Z.shape[0],
    )

    test_inputs = create_test_inputs(*mode_optimiser.dataset)
    # mogpe_plotter = QuadcopterPlotter(
    mogpe_plotter = Plotter2D(
        model=mode_optimiser.dynamics.mosvgpe,
        X=mode_optimiser.dataset[0],
        Y=mode_optimiser.dataset[1],
        test_inputs=test_inputs,
        # static=False,
    )

    train_dataset_tf, num_batches_per_epoch = create_tf_dataset(
        train_dataset, num_data=train_dataset[0].shape[0], batch_size=batch_size
    )
    test_dataset_tf, _ = create_tf_dataset(
        test_dataset, num_data=test_dataset[0].shape[0], batch_size=batch_size
    )

    # training_loss = mode_optimiser.dynamics.build_training_loss(
    #     train_dataset_tf, compile=compile_loss_fn
    # )

    fast_tasks = init_fast_tasks_bounds(
        log_dir,
        train_dataset_tf,
        mode_optimiser.dynamics.mosvgpe,
        test_dataset=test_dataset_tf,
        # training_loss=training_loss,
        fast_tasks_period=fast_tasks_period,
    )
    slow_tasks = mogpe_plotter.tf_monitor_task_group(
        log_dir,
        slow_period=slow_tasks_period
        # slow_tasks_period=slow_tasks_period,
    )
    monitor = Monitor(fast_tasks, slow_tasks)

    # Init checkpoint manager for saving model during training
    # ckpt = tf.train.Checkpoint(model=mode_optimiser)
    # manager = tf.train.CheckpointManager(ckpt, log_dir, max_to_keep=num_ckpts)
    manager = init_checkpoint_manager(
        model=mode_optimiser,
        log_dir=log_dir,
        num_ckpts=num_ckpts,
        mode_opt_gin_config=mode_opt_config,
        mogpe_toml_config=mogpe_config_file,
        train_dataset=train_dataset,
        test_dataset=test_dataset,
    )

    training_spec = ModeOptDynamicsTrainingSpec(
        num_epochs=num_epochs,
        batch_size=batch_size,
        # optimiser=optimiser,
        learning_rate=learning_rate,
        logging_epoch_freq=logging_epoch_freq,
        compile_loss_fn=compile_loss_fn,
        monitor=monitor,
        manager=manager,
    )

    mode_optimiser.optimise_dynamics(dataset=train_dataset, training_spec=training_spec)


if __name__ == "__main__":
    mode_opt_config = "./scenario_4/configs/learn_dynamics_subset_2_config.gin"
    # mode_opt_config = "./scenario_4/configs/learn_dynamics_subset_config.gin"
    # mode_opt_config = "./scenario_4/configs/learn_dynamics_initial_config.gin"
    gin.parse_config_files_and_bindings([mode_opt_config], None)

    train_dataset, test_dataset = load_vcpm_dataset()
    run_mode_opt_learn_dynamics(
        mode_opt_config=mode_opt_config,
        train_dataset=train_dataset,
        test_dataset=test_dataset,
    )
