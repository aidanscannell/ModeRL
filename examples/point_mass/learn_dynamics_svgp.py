#!/usr/bin/env python3
import pkg_resources
from datetime import datetime

import gpflow as gpf
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from gpflow.config import default_float
from gpflow.models import SVGP
from gpflow.monitor import ModelToTensorBoard, MonitorTaskGroup, ScalarToTensorBoard
from mogpe.training.training_loops import monitored_training_loop

# from mogpe.experts import SVGPExpert, SVGPExperts
# from mogpe.gating_networks import SVGPGatingNetwork
# from mogpe.gps.kernels import SigmoidRectangle
# from mogpe.helpers import Plotter2D
# from mogpe.mixture_of_experts import MixtureOfSVGPExperts
from mogpe.training.utils import create_tf_dataset, init_inducing_variables

# import simenvs
from simenvs.core import make
from simenvs.point_mass.plotter import Plotter2D

# Configure environment
env = make("point-mass/scenario-2")
# env = simenvs.make("point_mass/scenario-2")


def rollout(policy, timesteps):
    """Rollout a given policy on the environment

    :param policy: Callable representing policy to rollout
    :param timesteps: number of timesteps to rollout
    :returns: (X, Y)
        X: list of training inputs - (x, u) where x=state and u=control
        Y: list of training targets - differences y = x(t) - x(t-1)
    """
    x_set = []
    y_set = []
    env.reset()
    x, _, _, _ = env.step(0)
    # x = np.append(x, np.sin(x[1]))
    # x = np.append(x, np.cos(x[1]))
    # x = np.delete(x, 1)
    print(x)
    for t in range(timesteps):
        env.render()
        # print(x)
        u = policy(x)
        # print("u_%i: %.3f" % (t, u))
        x_new, reward, done, _ = env.step(u)
        if done:
            print("Episode finished after {} timesteps".format(t + 1))
            break
        # x_new = np.append(x_new, np.sin(x_new[1]))
        # x_new = np.append(x_new, np.cos(x_new[1]))
        # x_new = np.delete(x_new, 1)
        x_set.append(np.hstack((x, u)))
        y_set.append(x_new - x)
        x = x_new
    return np.stack(x_set), np.stack(y_set)


def random_policy(x):
    a = env.action_space.sample() * 4
    # a = env.action_space.sample()
    if a >= env.action_space.high:
        a = env.action_space.high
    if a <= env.action_space.low:
        a = env.action_space.low
    # print(a)
    return a


# config training
num_inducing = 90
num_samples = 1
batch_size = 300
num_epochs = 8000
logging_epoch_freq = 50
slow_tasks_period = 500
slow_tasks_period = 200
fast_tasks_period = 10
num_ckpts = 5

log_dir = "../logs/point_mass/scenario-2/learn/single-gp/" + datetime.now().strftime(
    "%m-%d-%H%M%S"
)

data_path = pkg_resources.resource_filename(
    "simenvs", "point_mass/scenario-2/data/point_mass_1000_4.npz"
)
print(data_path)
data = np.load(data_path)
print(data)
X = data["x"]
Y = data["y"]
print("X")
print(X.shape)
print(Y.shape)

# Load data set from npz file
# X = tf.convert_to_tensor(X[:, 0:2], dtype=default_float())
X = tf.convert_to_tensor(X, dtype=default_float())
Y = tf.convert_to_tensor(Y, dtype=default_float())
dataset = (X, Y)
output_dim = Y.shape[1]
input_dim = X.shape[1]
num_data = Y.shape[0]
train_dataset, num_batches_per_epoch = create_tf_dataset(dataset, num_data, batch_size)


# Initialise SVGP
noise_var = 1.9
lengthscales_1 = tf.Variable([0.5, 0.5], dtype=default_float())
lengthscales_1 = tf.Variable([0.5, 0.5, 0.5, 0.5], dtype=default_float())
variance_1 = tf.Variable([20.0], dtype=default_float())
# lengthscales_2 = tf.Variable([0.5, 0.5], dtype=default_float())
lengthscales_2 = tf.Variable([0.5, 0.5, 0.5, 0.5], dtype=default_float())
variance_2 = tf.Variable([20.0], dtype=default_float())
kern_1 = gpf.kernels.RBF(lengthscales=lengthscales_1, variance=variance_1)
kern_2 = gpf.kernels.RBF(lengthscales=lengthscales_2, variance=variance_2)
kern_list = [kern_1, kern_2]
kernel = gpf.kernels.SeparateIndependent(kern_list)
mean_function = gpf.mean_functions.Constant()
likelihood = gpf.likelihoods.Gaussian(noise_var)
inducing_variable = init_inducing_variables(X, num_inducing)
model = SVGP(
    kernel=kernel,
    likelihood=likelihood,
    inducing_variable=inducing_variable,
    mean_function=mean_function,
    num_latent_gps=output_dim,
    q_diag=False,
    whiten=True,
    # num_data=num_data,
)


gpf.utilities.print_summary(model)
training_loss = model.training_loss_closure(iter(train_dataset))

# configure training tasks (plotting etc)
plotter = Plotter2D(model, X, Y)
plotter.plot_model()
plt.show()

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
