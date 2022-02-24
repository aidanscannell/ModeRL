#!/usr/bin/env python3
import gin
import gpflow as gpf
import tensorflow as tf

from mogpe.training.utils import create_tf_dataset
from velocity_controlled_point_mass.utils import config_learn_dynamics


def learn_dynamics_with_single_expert(mode_opt, dataset, training_spec):
    optimiser = tf.optimizers.Adam(learning_rate=training_spec.learning_rate)
    train_dataset, num_batches_per_epoch = create_tf_dataset(
        dataset=dataset,
        num_data=dataset[0].shape[0],
        batch_size=training_spec.batch_size,
    )

    # Use standard SVGP ELBO for training
    training_loss = mode_opt.dynamics.mosvgpe.experts.experts_list[
        mode_opt.desired_mode
    ].training_loss_closure(iter(train_dataset))

    @tf.function
    def optimisation_step():
        optimiser.minimize(
            training_loss,
            mode_opt.dynamics.mosvgpe.experts.experts_list[
                mode_opt.desired_mode
            ].trainable_variables,
        )

    for epoch in range(training_spec.num_epochs):
        for _ in range(num_batches_per_epoch):
            optimisation_step()
        if training_spec.monitor is not None:
            training_spec.monitor(epoch)
        epoch_id = epoch + 1
        if epoch_id % training_spec.logging_epoch_freq == 0:
            tf.print(f"Epoch {epoch_id}: ELBO (train) {training_loss()}")
            if training_spec.manager is not None:
                training_spec.manager.save()


if __name__ == "__main__":
    mode_opt_config_file = "./velocity_controlled_point_mass/scenario_5/configs/learn_dynamics_svgp_config.gin"
    gin.parse_config_files_and_bindings([mode_opt_config_file], None)

    mode_optimiser, training_spec, train_dataset = config_learn_dynamics(
        mode_opt_config_file=mode_opt_config_file
    )
    gpf.utilities.print_summary(mode_optimiser)

    learn_dynamics_with_single_expert(
        mode_optimiser, dataset=train_dataset, training_spec=training_spec
    )
