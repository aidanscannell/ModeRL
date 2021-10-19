#!/usr/bin/env python3
import tensorflow as tf


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
