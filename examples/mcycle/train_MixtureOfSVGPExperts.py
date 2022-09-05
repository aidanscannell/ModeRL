#!/usr/bin/env python3
from typing import Callable, List, Optional

import gpflow as gpf
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from gpflow import default_float
from gpflow.inducing_variables import InducingPoints
from gpflow.kernels import RBF
from gpflow.likelihoods import Gaussian
from gpflow.mean_functions import Constant
from mosvgpe.custom_types import Dataset, InputData
from mosvgpe.experts import SVGPExpert
from mosvgpe.gating_networks import SVGPGatingNetwork
from mosvgpe.mixture_of_experts import MixtureOfSVGPExperts

# from mosvgpe.utils import sample_mosvgpe_inducing_inputs_from_data
from wandb.keras import WandbCallback

import wandb

PlotFn = Callable[[], matplotlib.figure.Figure]

colors = ["m", "c"]

tf.random.set_seed(42)
np.random.seed(42)


def build_SVGPExpert(
    X: InputData,
    lengthscales: List[float] = [1.0],
    kernel_variance: float = 1.0,
    noise_variance: float = 1.0,
) -> SVGPExpert:
    likelihood = Gaussian(variance=noise_variance)
    mean_function = Constant(c=0.0)
    kernel = RBF(lengthscales=lengthscales, variance=kernel_variance)
    idx = np.random.choice(range(X.shape[0]), size=num_inducing, replace=False)
    Z = X.numpy()[idx, ...].reshape(-1, X.shape[1])
    inducing_variable = InducingPoints(Z)
    return SVGPExpert(
        kernel=kernel,
        likelihood=likelihood,
        mean_function=mean_function,
        inducing_variable=inducing_variable,
        num_latent_gps=1,
        q_diag=False,
        q_mu=None,
        q_sqrt=None,
        whiten=True,
    )


def build_SVGPGatingNetwork(
    X: InputData,
    lengthscales: List[float] = [1.0],
    kernel_variance: float = 1.0,
    num_gating_gps: int = 1,
) -> SVGPGatingNetwork:
    if num_gating_gps == 1:
        kernel = RBF(lengthscales=lengthscales, variance=kernel_variance)
        idx = np.random.choice(range(X.shape[0]), size=num_inducing, replace=False)
        Z = X.numpy()[idx, ...].reshape(-1, X.shape[1])
        inducing_variable = InducingPoints(Z)
    elif num_gating_gps > 1:
        kernels = [
            RBF(lengthscales=lengthscales, variance=kernel_variance)
            for _ in range(num_gating_gps)
        ]
        kernel = gpf.kernels.SeparateIndependent(kernels=kernels)
        raise NotImplementedError(
            "need to implement method for instantiating multioutput inducing variables"
        )
    else:
        raise AttributeError("num_gating_gps should be >= 1")
    return SVGPGatingNetwork(
        kernel=kernel,
        inducing_variable=inducing_variable,
        mean_function=None,  # Use zero mean function
        q_diag=False,
        q_mu=None,
        q_sqrt=None,
        whiten=True,
    )


# def build_MixtureOfSVGPExperts(num_experts: int = 2, num_gating_gps: int = 1):
#     experts_list = [build_SVGPExpert() for _ in range(num_experts)]
#     gating_network = build_SVGPGatingNetwork(num_gating_gps)
#     return MixtureOfSVGPExperts(
#         experts_list=experts_list, gating_network=gating_network
#     )
#
def build_MixtureOfSVGPExperts(num_experts: int = 2, num_gating_gps: int = 1):
    """Build model and sample GP inducing variables from X"""
    experts_list = []
    for k in range(num_experts):
        expert = build_SVGPExpert(
            X=X,
            lengthscales=experts_lengthscales[k],
            kernel_variance=experts_kernel_variances[k],
        )
        experts_list.append(expert)
    gating_network = build_SVGPGatingNetwork(X=X, num_gating_gps=num_gating_gps)
    model = MixtureOfSVGPExperts(
        experts_list=experts_list, gating_network=gating_network
    )
    # sample_mosvgpe_inducing_inputs_from_data(X, model)
    gpf.utilities.print_summary(model)
    model(X)  # Needed to build with shapes


def load_mcycle_data(filename: str = "./mcycle.csv") -> Dataset:
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


class PlottingCallback(tf.keras.callbacks.Callback):
    def __init__(self, plot_fn: PlotFn, logging_epoch_freq: int = 10, name: str = ""):
        self.plot_fn = plot_fn
        self.logging_epoch_freq = logging_epoch_freq
        self.name = name

    def on_epoch_end(self, epoch: int, logs=None):
        if epoch % self.logging_epoch_freq == 0:
            fig = self.plot_fn()
            wandb.log({self.name: wandb.Image(fig)})
            # wandb.log({self.name: fig})


def plot_experts_gps():
    fig = plt.figure()
    gs = fig.add_gridspec(1, 1)
    ax = gs.subplots()
    for k, expert in enumerate(model.experts_list):
        mean, var = expert.predict_f(test_inputs)
        # f_samples = expert.predict_f_samples(test_inputs, 5)
        Z = expert.gp.inducing_variable.Z
        # for f_sample in f_samples:
        #     axs[row].plot(self.test_inputs, f_sample, alpha=0.3)
        plot_gp(ax, mean[:, 0], var[:, 0], label="k=" + str(k + 1), color=colors[k])
        ax.scatter(Z, np.zeros(Z.shape), marker="|", color=colors[k])
    ax.legend()
    return fig


def plot_gating_networks_gps():
    fig = plt.figure()
    gs = fig.add_gridspec(1, 1)
    ax = gs.subplots()
    mean, var = model.gating_network.predict_h(test_inputs)
    for k in range(mean.shape[-1]):
        plot_gp(ax, mean[:, k], var[:, k], label="k=" + str(k + 1), color=colors[k])
        Z = model.gating_network.gp.inducing_variable.Z
        ax.scatter(Z, np.zeros(Z.shape), marker="|", color=colors[k])
    ax.legend()
    return fig


def plot_mixing_probs():
    fig = plt.figure()
    gs = fig.add_gridspec(1, 1)
    ax = gs.subplots()
    probs = model.gating_network.predict_mixing_probs(test_inputs)
    for k in range(model.num_experts):
        ax.plot(test_inputs, probs[:, k], label="k=" + str(k + 1), color=colors[k])
    ax.legend()
    return fig


def plot_gp(ax, mean, var, label="", color="C0", alpha=0.4):
    ax.scatter(X, Y, marker="x", color="k", alpha=alpha)
    ax.plot(test_inputs, mean, color=color, lw=2, label=label)
    ax.fill_between(
        test_inputs[:, 0],
        mean - 1.96 * np.sqrt(var),
        mean + 1.96 * np.sqrt(var),
        color=color,
        alpha=0.2,
    )


if __name__ == "__main__":
    mcylce_dataset_filename = "./mcycle.csv"
    num_inducing = 32
    num_experts = 2
    num_gating_gps = 1
    experts_lengthscales = [1.0, 10.0]
    experts_kernel_variances = [1.0, 1.0]

    batch_size = 16
    epochs = 10000
    # epochs = 25000
    num_samples = 1
    # learning_rate = 1e-3
    learning_rate = 1e-2
    logging_epoch_freq = 100
    verbose = True
    validation_split = 0.2

    # Initialise WandB run and save experiment config
    wandb.init(project="mosvgpe", entity="aidanscannell")
    wandb.config = {
        "num_inducing": num_inducing,
        "num_experts": num_experts,
        "num_gating_gps": num_gating_gps,
        "experts_lengthscales": experts_lengthscales,
        "experts_kernel_variances": experts_kernel_variances,
        "batch_size": batch_size,
        "epochs": epochs,
        "num_samples": num_samples,
        "learning_rate": learning_rate,
        "validation_split": validation_split,
    }

    # Load the dataset
    X, Y = load_mcycle_data(mcylce_dataset_filename)

    # Build model and sample GP inducing variables from X
    experts_list = []
    for k in range(num_experts):
        expert = build_SVGPExpert(
            X=X,
            lengthscales=experts_lengthscales[k],
            kernel_variance=experts_kernel_variances[k],
        )
        experts_list.append(expert)
    gating_network = build_SVGPGatingNetwork(X=X, num_gating_gps=num_gating_gps)
    model = MixtureOfSVGPExperts(
        experts_list=experts_list, gating_network=gating_network
    )
    # sample_mosvgpe_inducing_inputs_from_data(X, model)
    gpf.utilities.print_summary(model)
    model(X)  # Needed to build with shapes

    # Define WandbCallback for experiment tracking and plotting callbacks
    test_inputs = np.linspace(
        tf.reduce_min(X) * 1.2, tf.reduce_max(X) * 1.2, 100
    ).reshape(-1, 1)
    callbacks = [
        WandbCallback(
            monitor="val_loss",
            log_weights=False,
            log_evaluation=True,
            validation_steps=5,
        ),
        PlottingCallback(
            plot_experts_gps, logging_epoch_freq=logging_epoch_freq, name="Expert GPs"
        ),
        PlottingCallback(
            plot_gating_networks_gps,
            logging_epoch_freq=logging_epoch_freq,
            name="Gating Network GPs",
        ),
        PlottingCallback(
            plot_mixing_probs,
            logging_epoch_freq=logging_epoch_freq,
            name="Mixing Probs",
        ),
    ]

    # Compile the Keras model and train it
    optimiser = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimiser)

    model.fit(
        X,
        Y,
        callbacks=callbacks,
        batch_size=batch_size,
        epochs=epochs,
        verbose=verbose,
        validation_split=validation_split,
    )
