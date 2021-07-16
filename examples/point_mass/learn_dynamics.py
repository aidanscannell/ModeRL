#!/usr/bin/env python3
from datetime import datetime

import gpflow as gpf
import matplotlib.pyplot as plt
import numpy as np
import pkg_resources
import tensorflow as tf
from gpflow.config import default_float
from gpflow.monitor import ModelToTensorBoard, MonitorTaskGroup, ScalarToTensorBoard
from mogpe.experts import SVGPExpert, SVGPExperts
from mogpe.gating_networks import SVGPGatingNetwork
from mogpe.gps.kernels import SigmoidRectangle
from mogpe.helpers import Plotter2D

# from simenvs.point_mass.plotter import Plotter2D
from mogpe.mixture_of_experts import MixtureOfSVGPExperts
from mogpe.training.training_loops import monitored_training_loop
from mogpe.training.utils import create_tf_dataset, init_inducing_variables

# import simenvs
from simenvs.core import make


class PointMassMeanFunction(gpf.mean_functions.MeanFunction):
    def __init__(self, delta_time=0.05):
        self.delta_time = delta_time

    def __call__(self, X):
        # state = X[:, 0:2]
        control = X[:, 2:4]
        velocity = control
        return velocity * self.delta_time


def init_inducing_dist(num_inducing, output_dim, q_mu=0.0, q_sqrt=1.0, q_mu_var=2.0):
    q_mu = (
        q_mu * np.ones((num_inducing, output_dim))
        + np.random.randn(num_inducing, output_dim) * q_mu_var
    )
    q_sqrt = np.array(
        [
            q_sqrt * np.eye(num_inducing, dtype=default_float())
            for _ in range(output_dim)
        ]
    )
    return q_mu, q_sqrt


def create_model(X, output_dim=2, num_inducing=150, num_samples=1):
    if not isinstance(X, np.ndarray):
        X = X.numpy()
    num_data = X.shape[0]
    # input_dim = X.shape[1]
    # initialise expert 1
    noise_var_1 = 1.9
    lengthscales_11 = tf.Variable([0.5, 0.5], dtype=default_float())
    lengthscales_11 = tf.Variable([0.5, 0.5], dtype=default_float())
    lengthscales_11 = tf.Variable([0.5, 0.5, 0.5, 0.5], dtype=default_float())
    variance_11 = tf.Variable([20.0], dtype=default_float())
    lengthscales_12 = tf.Variable([0.5, 0.5, 0.5, 0.5], dtype=default_float())
    variance_12 = tf.Variable([20.0], dtype=default_float())
    kern_11 = gpf.kernels.RBF(lengthscales=lengthscales_11, variance=variance_11)
    kern_12 = gpf.kernels.RBF(lengthscales=lengthscales_12, variance=variance_12)
    kern_list_1 = [kern_11, kern_12]
    kernel_1 = gpf.kernels.SeparateIndependent(kern_list_1)
    c = np.array([0.0, -0.2])
    mean_function_1 = gpf.mean_functions.Constant(c=c)
    mean_function_1 += PointMassMeanFunction()
    likelihood_1 = gpf.likelihoods.Gaussian(noise_var_1)
    inducing_variable_1 = init_inducing_variables(X, num_inducing)
    q_mu, q_sqrt = init_inducing_dist(
        num_inducing, output_dim, q_mu=0.0, q_sqrt=1.0, q_mu_var=2.0
    )
    expert_1 = SVGPExpert(
        kernel_1,
        likelihood_1,
        inducing_variable_1,
        mean_function_1,
        num_latent_gps=output_dim,
        q_mu=q_mu,
        q_sqrt=q_sqrt,
        q_diag=False,
        whiten=True,
        num_data=num_data,
    )

    # initialise expert 2
    noise_var_2 = 0.0011
    # lengthscales_21 = tf.Variable([10.0, 10.0], dtype=default_float())
    lengthscales_21 = tf.Variable([10.0, 10.0, 10.0, 10.0], dtype=default_float())
    variance_21 = tf.Variable([0.1], dtype=default_float())
    lengthscales_22 = tf.Variable([10.0, 10.0], dtype=default_float())
    lengthscales_22 = tf.Variable([10.0, 10.0, 10.0, 10.0], dtype=default_float())
    variance_22 = tf.Variable([0.1], dtype=default_float())
    kern_21 = gpf.kernels.RBF(lengthscales=lengthscales_21, variance=variance_21)
    kern_22 = gpf.kernels.RBF(lengthscales=lengthscales_22, variance=variance_22)
    kern_list_2 = [kern_21, kern_22]
    kernel_2 = gpf.kernels.SeparateIndependent(kern_list_2)
    c = np.array([0.0, 0.0])
    mean_function_2 = gpf.mean_functions.Constant(c=c)
    mean_function_2 *= PointMassMeanFunction()
    likelihood_2 = gpf.likelihoods.Gaussian(noise_var_2)
    inducing_variable_2 = init_inducing_variables(X, num_inducing)
    q_mu, q_sqrt = init_inducing_dist(
        num_inducing, output_dim, q_mu=0.0, q_sqrt=1.0, q_mu_var=2.0
    )
    expert_2 = SVGPExpert(
        kernel_2,
        likelihood_2,
        inducing_variable_2,
        mean_function_2,
        num_latent_gps=output_dim,
        q_mu=q_mu,
        q_sqrt=q_sqrt,
        q_diag=False,
        whiten=True,
        num_data=num_data,
    )

    # initialise experts
    experts = SVGPExperts([expert_1, expert_2])

    # initialise gating function
    lengthscales_gating = tf.Variable([0.5, 0.5], dtype=default_float())
    # lengthscales_gating = tf.Variable([0.5, 0.5, 0.5, 0.5], dtype=default_float())
    variance_gating = tf.Variable([10.0], dtype=default_float())
    kernel_gating = gpf.kernels.RBF(
        lengthscales=lengthscales_gating, variance=variance_gating, active_dims=[0, 1]
    )
    # # center = tf.Variable([1.5, 2.0], dtype=default_float())
    # width = tf.Variable([1.0, 2.0], dtype=default_float())
    # variance_gating = np.array([10.0])
    # # center = np.array([1.5, 2.0])
    # center = np.array([0.7, 1.2])
    # width = np.array([1.0, 2.0])
    # kernel_gating = SigmoidRectangle(width=width, center=center, variance=variance_gating)
    # set_trainable(kernel_gating.width, False)
    # set_trainable(kernel_gating.center, False)
    # set_trainable(kernel_gating.variance, False)
    mean_function_gating = gpf.mean_functions.Zero()
    likelihood_gating = gpf.likelihoods.Bernoulli()
    idx = np.random.choice(range(X.shape[0]), size=num_inducing, replace=False)
    inducing_inputs = X[idx, 0:2]
    # inducing_inputs = X[idx, 0:2].reshape(num_inducing, input_dim)
    inducing_variable_gating = gpf.inducing_variables.InducingPoints(inducing_inputs)
    q_mu, q_sqrt = init_inducing_dist(
        num_inducing, output_dim=1, q_mu=0.0, q_sqrt=10.0, q_mu_var=2.0
    )
    gating_network = SVGPGatingNetwork(
        kernel_gating,
        likelihood_gating,
        inducing_variable_gating,
        mean_function_gating,
        num_gating_functions=1,
        q_mu=q_mu,
        q_sqrt=q_sqrt,
        q_diag=False,
        whiten=True,
        num_data=num_data,
    )

    # initialise model and training loss
    model = MixtureOfSVGPExperts(
        gating_network=gating_network,
        experts=experts,
        num_samples=num_samples,
        num_data=num_data,
    )
    gpf.utilities.print_summary(model)
    return model


if __name__ == "__main__":
    # Configure environment
    # env = make("point-mass/scenario-2")
    scenario = "scenario-1"
    scenario = "scenario-3"
    scenario = "scenario-4"
    env = make("point-mass/" + scenario)
    # env = simenvs.make("point_mass/scenario-2")
    log_dir = (
        "../logs/point_mass/"
        + scenario
        + "/learn/two_experts/"
        + datetime.now().strftime("%m-%d-%H%M%S")
    )

    # model params
    num_inducing = 150
    num_samples = 1

    # training params
    batch_size = 64
    num_epochs = 8000
    slow_tasks_period = 100
    fast_tasks_period = 10
    logging_epoch_freq = 50
    num_ckpts = 5

    # Load data set from npz file
    data_path = pkg_resources.resource_filename(
        "simenvs",
        "point_mass/" + scenario + "/data/point_mass_random_10000.npz"
        # "simenvs", "point_mass/" + scenario + "/data/point_mass_1000_4.npz"
    )
    data = np.load(data_path)
    X = data["x"]
    Y = data["y"]
    output_dim = Y.shape[1]
    input_dim = X.shape[1]
    num_data = Y.shape[0]
    X = tf.convert_to_tensor(X, dtype=default_float())
    Y = tf.convert_to_tensor(Y, dtype=default_float())
    print("X: {}, Y: {}".format(X.shape, Y.shape))
    dataset = (X, Y)
    train_dataset, num_batches_per_epoch = create_tf_dataset(
        dataset, num_data, batch_size
    )

    # Initialise model
    model = create_model(
        X, output_dim=output_dim, num_inducing=num_inducing, num_samples=num_samples
    )
    training_loss = model.training_loss_closure(iter(train_dataset))

    # Configure training tasks (plotting etc)
    # plotter = Plotter2D(model.experts.experts_list[0], X, Y)
    num_test = 400
    factor = 1.2
    sqrtN = int(np.sqrt(num_test))
    xx = np.linspace(
        tf.reduce_min(X[:, 0]) * factor, tf.reduce_max(X[:, 0]) * factor, sqrtN
    )
    yy = np.linspace(
        tf.reduce_min(X[:, 1]) * factor, tf.reduce_max(X[:, 1]) * factor, sqrtN
    )
    xx, yy = np.meshgrid(xx, yy)
    test_inputs = np.column_stack([xx.reshape(-1), yy.reshape(-1)])
    test_inputs = np.concatenate(
        [
            test_inputs,
            np.ones((test_inputs.shape[0], 2)),
        ],
        -1,
    )
    plotter = Plotter2D(model, X, Y, test_inputs=test_inputs)
    # plotter.plot_model()
    # plt.show()
    slow_tasks = plotter.tf_monitor_task_group(log_dir, slow_tasks_period)
    elbo_task = ScalarToTensorBoard(log_dir, training_loss, "elbo")
    model_task = ModelToTensorBoard(log_dir, model)
    fast_tasks = MonitorTaskGroup([model_task, elbo_task], period=fast_tasks_period)

    ckpt = tf.train.Checkpoint(model=model)
    manager = tf.train.CheckpointManager(ckpt, log_dir, max_to_keep=num_ckpts)

    monitored_training_loop(
        model,
        training_loss,
        epochs=num_epochs,
        fast_tasks=fast_tasks,
        slow_tasks=slow_tasks,
        num_batches_per_epoch=num_batches_per_epoch,
        logging_epoch_freq=logging_epoch_freq,
        manager=manager,
    )
