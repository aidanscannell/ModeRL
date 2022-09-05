import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from mosvgpe.custom_types import InputData
from mosvgpe.mixture_of_experts import MixtureOfSVGPExperts

# test_inputs = tf.reshape(
#     np.linspace(
#         tf.reduce_min(X) * factor,
#         tf.reduce_max(X) * factor,
#         num_test,
#     ),
#     [num_test, tf.shape(X)[1]],
# )


# def plot_samples(self, fig, ax, input_broadcast, y_samples, color=color_3):
#     ax.scatter(
#         input_broadcast,
#         y_samples,
#         marker=".",
#         s=4.9,
#         color=color,
#         lw=0.4,
#         rasterized=True,
#         alpha=0.2,
#     )


# def plot_y(self, fig, ax):
#     # tf.print("Plotting y...")
#     alpha = 0.4
#     ax.scatter(self.X, self.Y, marker="x", color="k", alpha=alpha)
#     y_dist = self.model.predict_y(self.test_inputs)
#     y_samples = y_dist.sample(self.num_samples)
#     ax.plot(self.test_inputs, y_dist.mean(), color="k")

#     self.test_inputs_broadcast = np.expand_dims(self.test_inputs, 0)

#     for i in range(self.num_samples):
#         self.plot_samples(fig, ax, self.test_inputs_broadcast, y_samples[i, :, :])


# def plot_gp(ax, mean, var, label="", alpha=0.4):
#     ax.scatter(X, Y, marker="x", color="k", alpha=alpha)
#     ax.plot(test_inputs, mean, "C0", lw=2, label=label)
#     ax.fill_between(
#         test_inputs[:, 0],
#         mean - 1.96 * np.sqrt(var),
#         mean + 1.96 * np.sqrt(var),
#         color="C0",
#         alpha=0.2,
#     )
