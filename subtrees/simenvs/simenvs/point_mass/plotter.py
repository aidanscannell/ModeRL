#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
import palettable
import tensorflow as tf
from gpflow.monitor import ImageToTensorBoard, MonitorTaskGroup
from mogpe.training.monitor import ImageWithCbarToTensorBoard
from mpl_toolkits.axes_grid1 import make_axes_locatable


color_1 = "olive"
color_2 = "darkmagenta"
color_2 = "darkslategrey"
color_3 = "darkred"
color_3 = "lime"
color_obs = "red"


class Plotter2D:
    def __init__(
        self,
        model,
        X,
        Y,
        test_inputs=None,
        num_samples=100,
        params=None,
        num_levels=10,
        cmap=palettable.scientific.sequential.Bilbao_15.mpl_colormap,
    ):
        # super().__init__(model, X, Y, num_samples, params)
        self.model = model
        self.X = X
        self.Y = Y
        self.input_dim = X.shape[1]
        self.output_dim = Y.shape[1]
        self.num_samples = num_samples
        self.params = params
        self.cmap = cmap
        self.num_levels = num_levels
        # self.levels = np.linspace(0., 1., num_levels)
        self.levels = np.linspace(0.0, 1.0, 50)
        if test_inputs is None:
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
            self.test_inputs = np.column_stack([xx.reshape(-1), yy.reshape(-1)])
            print("self.test_inputs")
            print(self.test_inputs.shape)
            self.test_inputs = np.concatenate(
                [
                    self.test_inputs,
                    np.ones(
                        (
                            self.test_inputs.shape[0],
                            self.input_dim - self.test_inputs.shape[1],
                        )
                    ),
                ],
                -1,
            )
            print(self.test_inputs.shape)
        else:
            self.test_inputs = test_inputs

    def plot_gp(self, fig, axs, mean, var, mean_levels=None, var_levels=None):
        """Plots contours and colorbars for mean and var side by side

        :param axs: [2,]
        :param mean: [num_data, 1]
        :param var: [num_data, 1]
        :param mean_levels: levels for mean contourf e.g. np.linspace(0, 1, 10)
        :param var_levels: levels for var contourf e.g. np.linspace(0, 1, 10)
        """
        mean_contf, var_contf = self.plot_gp_contf(
            fig, axs, mean, var, mean_levels, var_levels
        )
        mean_cbar = self.cbar(fig, axs[0], mean_contf)
        var_cbar = self.cbar(fig, axs[1], var_contf)
        return np.array([mean_cbar, var_cbar])

    def plot_gp_contf(self, fig, axs, mean, var, mean_levels=None, var_levels=None):
        """Plots contours for mean and var side by side

        :param axs: [2,]
        :param mean: [num_data, 1]
        :param var: [num_data, 1]
        :param mean_levels: levels for mean contourf e.g. np.linspace(0, 1, 10)
        :param var_levels: levels for var contourf e.g. np.linspace(0, 1, 10)
        """
        mean_contf = axs[0].tricontourf(
            self.test_inputs[:, 0],
            self.test_inputs[:, 1],
            mean,
            100,
            levels=mean_levels,
            cmap=self.cmap,
        )
        var_contf = axs[1].tricontourf(
            self.test_inputs[:, 0],
            self.test_inputs[:, 1],
            var,
            100,
            levels=var_levels,
            cmap=self.cmap,
        )
        return mean_contf, var_contf

    def cbar(self, fig, ax, contf):
        """Adds cbar to ax or ax[0] is np.ndarray

        :param ax: either a matplotlib ax or np.ndarray axs
        :param contf: contourf
        """
        if isinstance(ax, np.ndarray):
            divider = make_axes_locatable(ax[0])
        else:
            divider = make_axes_locatable(ax)
        cax = divider.append_axes("top", size="5%", pad=0.05)
        cbar = fig.colorbar(
            contf,
            ax=ax,
            use_gridspec=True,
            cax=cax,
            format="%0.2f",
            orientation="horizontal",
        )
        cax.xaxis.set_ticks_position("top")
        cax.xaxis.set_label_position("top")
        return cbar

    def plot_gps_shared_cbar(self, fig, axs, means, vars):
        """Plots mean and var for each expert in each output dim

        The rows iterate through experts and then output_dim
        e.g. row 1 = expert 1, output_dim 1
             row 2 = expert 1, output_dim 2
             row 3 = expert 2, output_dim 1
             row 4 = expert 2, output_dim 2

        :param axs: [num_experts*output_dim, 2]
        :param means: [num_data, output_dim, num_experts]
        :param vars: [num_data, output_dim, num_experts]
        """
        # output_dim = tf.shape(means)[1]
        output_dim = means.shape[1]
        # mean_levels = tf.linspace(
        #     tf.math.reduce_min(means), tf.math.reduce_max(means), self.num_levels
        # )
        # var_levels = tf.linspace(
        #     tf.math.reduce_min(vars), tf.math.reduce_max(vars), self.num_levels
        # )
        row = 0
        for j in range(output_dim):
            # if row != num_experts * output_dim - 1:
            #     axs[row, 0].get_xaxis().set_visible(False)
            #     axs[row, 1].get_xaxis().set_visible(False)
            mean_contf, var_contf = self.plot_gp_contf(
                fig,
                axs[j, :],
                means[:, j],
                vars[:, j],
                # mean_levels=mean_levels,
                # var_levels=var_levels,
            )
        mean_cbar = self.cbar(fig, axs[:, 0], mean_contf)
        var_cbar = self.cbar(fig, axs[:, 1], var_contf)
        return np.array([mean_cbar, var_cbar])

    def plot_f(self, fig, axs):
        """Plots mean and var of moment matched predictive posterior

        :param axs: [output_dim, 2]
        """
        tf.print("Plotting experts f...")
        means, vars = self.model.predict_f(self.test_inputs)
        print("means")
        print(means.shape)
        print(vars.shape)
        if self.output_dim > 1:
            return self.plot_gps_shared_cbar(fig, axs, means, vars)
        else:
            return self.plot_gp(fig, axs, means[:, 0], vars[:, 0])

    def plot_y(self, fig, axs):
        """Plots mean and var of moment matched predictive posterior

        :param axs: [output_dim, 2]
        """
        tf.print("Plotting y (moment matched)...")
        means, vars = self.model.predict_y(self.test_inputs)
        if self.output_dim > 1:
            return self.plot_gps_shared_cbar(fig, axs, means, vars)
        else:
            return self.plot_gp(fig, axs, means[:, 0], vars[:, 0])

    def plot_model(self):
        fig, axs = plt.subplots(self.output_dim, 2)
        self.plot_f(fig, axs)
        fig, axs = plt.subplots(self.output_dim, 2)
        self.plot_y(fig, axs)

    #     nrows = self.num_experts * self.output_dim
    #     fig, ax = plt.subplots(self.num_experts, self.output_dim)
    #     self.plot_gating_network(fig, ax)
    #     if self.num_experts > 2:
    #         num_gating_gps = self.num_experts
    #     else:
    #         num_gating_gps = 1
    #     num_gating_gps *= self.output_dim
    #     fig, axs = plt.subplots(num_gating_gps, 2)
    #     self.plot_gating_gps(fig, axs)
    #     fig, axs = plt.subplots(nrows, 2, figsize=(10, 4))
    #     self.plot_experts_f(fig, axs)
    #     fig, axs = plt.subplots(nrows, 2, figsize=(10, 4))
    #     self.plot_experts_y(fig, axs)
    #     fig, axs = plt.subplots(self.output_dim, 2, figsize=(10, 4))
    #     self.plot_y(fig, axs)

    def tf_monitor_task_group(self, log_dir, slow_period=500):
        ncols = 2
        image_task_f = ImageWithCbarToTensorBoard(
            log_dir,
            self.plot_f,
            name="predict_f",
            fig_kw={"figsize": (10, 4)},
            subplots_kw={"nrows": self.output_dim, "ncols": ncols},
        )
        image_task_y = ImageWithCbarToTensorBoard(
            log_dir,
            self.plot_y,
            name="predict_y",
            fig_kw={"figsize": (10, 4)},
            subplots_kw={"nrows": self.output_dim, "ncols": ncols},
        )
        image_tasks = [image_task_f, image_task_y]
        return MonitorTaskGroup(image_tasks, period=slow_period)
