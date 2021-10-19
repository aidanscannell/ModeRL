#!/usr/bin/env python3
from typing import Callable, Tuple

import gin
import tensorflow as tf
from modeopt.dynamics import ModeOptDynamics
from mogpe.training import MixtureOfSVGPExperts_from_toml
from mogpe.training.utils import update_model_from_checkpoint


@gin.configurable
def init_ModeOptDynamics_from_mogpe_ckpt(
    mogpe_config_file: str,
    dataset: Tuple,
    mogpe_ckpt_dir: str = None,
    nominal_dynamics: Callable = None,
    desired_mode: int = 0,
    optimiser: tf.optimizers.Optimizer = tf.optimizers.Adam(),
):
    X, Y = dataset
    state_dim = Y.shape[1]
    control_dim = X.shape[1] - state_dim
    model = MixtureOfSVGPExperts_from_toml(mogpe_config_file, dataset=(X, Y))

    if mogpe_ckpt_dir is not None:
        model = update_model_from_checkpoint(model, mogpe_ckpt_dir)
    dynamics = ModeOptDynamics(
        mosvgpe=model,
        desired_mode=desired_mode,
        state_dim=state_dim,
        control_dim=control_dim,
        nominal_dynamics=nominal_dynamics,
        optimiser=optimiser,
    )
    return dynamics


def create_tf_dataset(dataset, batch_size):
    X, Y = dataset
    assert X.shape[0] == Y.shape[0]
    num_data = X.shape[0]
    prefetch_size = tf.data.experimental.AUTOTUNE
    shuffle_buffer_size = num_data // 2
    num_batches_per_epoch = num_data // batch_size
    train_dataset = tf.data.Dataset.from_tensor_slices(dataset)
    train_dataset = (
        train_dataset.repeat()
        .prefetch(prefetch_size)
        .shuffle(buffer_size=shuffle_buffer_size)
        .batch(batch_size, drop_remainder=True)
    )
    return train_dataset, num_batches_per_epoch
