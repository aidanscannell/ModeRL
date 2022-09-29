#!/usr/bin/env python3
import os

import numpy as np
import tensorflow as tf
from gpflow import default_float
from hydra.utils import get_original_cwd
from mosvgpe.custom_types import Dataset


def load_mcycle_data(filename: str = "./mcycle/mcycle.csv") -> Dataset:
    filename = os.path.join(get_original_cwd(), filename)
    data = np.loadtxt(filename, delimiter=",", skiprows=1, usecols=(1, 2))
    X = data[:, 0].reshape(-1, 1)
    Y = data[:, 1].reshape(-1, 1)
    print("Input data shape: ", X.shape)
    print("Output data shape: ", Y.shape)

    # standardise input
    X = (X - X.mean()) / X.std()
    Y = (Y - Y.mean()) / Y.std()

    X = tf.convert_to_tensor(X, dtype=default_float())
    Y = tf.convert_to_tensor(Y, dtype=default_float())
    return (X, Y)
