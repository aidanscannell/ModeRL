#!/usr/bin/env python3
import tensorflow as tf
from moderl.dynamics import ModeRLDynamics

import wandb


class DynamicsLoggingCallback(tf.keras.callbacks.Callback):
    def __init__(
        self, dynamics: ModeRLDynamics, logging_epoch_freq: int = 10, name: str = ""
    ):
        self.dynamics = dynamics
        self.logging_epoch_freq = logging_epoch_freq
        self.name = name

    # def on_epoch_end(self, epoch: int, logs=None):
    #     if epoch % self.logging_epoch_freq == 0:
    #         for k, expert in enumerate(self.dynamics.mosvgpe.experts_list):
    #             wandb.log(
    #                 {
    #                     "expert_{}_noise_var_1".format(
    #                         k
    #                     ): expert.gp.likelihood.variance[0]
    #                 }
    #             )

    def on_epoch_end(self, epoch: int, logs=None):
        if epoch % self.logging_epoch_freq == 0:
            wandb.log(
                {
                    "expert_1_noise_var_1": self.dynamics.mosvgpe.experts_list[
                        0
                    ].gp.likelihood.variance[0]
                }
            )
            wandb.log(
                {
                    "expert_1_noise_var_2": self.dynamics.mosvgpe.experts_list[
                        0
                    ].gp.likelihood.variance[1]
                }
            )
            wandb.log(
                {
                    "expert_2_noise_var_1": self.dynamics.mosvgpe.experts_list[
                        1
                    ].gp.likelihood.variance[0]
                }
            )
            wandb.log(
                {
                    "expert_2_noise_var_2": self.dynamics.mosvgpe.experts_list[
                        1
                    ].gp.likelihood.variance[1]
                }
            )
            wandb.log(
                {
                    "expert_1_kernel_var_1": self.dynamics.mosvgpe.experts_list[0]
                    .gp.kernel.kernels[0]
                    .variance.numpy()
                }
            )
            wandb.log(
                {
                    "expert_1_kernel_var_2": self.dynamics.mosvgpe.experts_list[0]
                    .gp.kernel.kernels[1]
                    .variance.numpy()
                }
            )
            wandb.log(
                {
                    "expert_2_kernel_var_1": self.dynamics.mosvgpe.experts_list[1]
                    .gp.kernel.kernels[0]
                    .variance.numpy()
                }
            )
            wandb.log(
                {
                    "expert_2_kernel_var_2": self.dynamics.mosvgpe.experts_list[1]
                    .gp.kernel.kernels[1]
                    .variance.numpy()
                }
            )
            wandb.log(
                {
                    "expert_1_kernel_lengthscale_1": self.dynamics.mosvgpe.experts_list[
                        0
                    ]
                    .gp.kernel.kernels[0]
                    .lengthscales.numpy()[0]
                }
            )
            wandb.log(
                {
                    "expert_1_kernel_lengthscale_2": self.dynamics.mosvgpe.experts_list[
                        0
                    ]
                    .gp.kernel.kernels[0]
                    .lengthscales.numpy()[1]
                }
            )
            wandb.log(
                {
                    "expert_1_kernel_lengthscale_3": self.dynamics.mosvgpe.experts_list[
                        0
                    ]
                    .gp.kernel.kernels[0]
                    .lengthscales.numpy()[2]
                }
            )
            wandb.log(
                {
                    "expert_1_kernel_lengthscale_4": self.dynamics.mosvgpe.experts_list[
                        0
                    ]
                    .gp.kernel.kernels[0]
                    .lengthscales.numpy()[3]
                }
            )
            wandb.log(
                {
                    "expert_2_kernel_lengthscale_1": self.dynamics.mosvgpe.experts_list[
                        1
                    ]
                    .gp.kernel.kernels[0]
                    .lengthscales.numpy()[0]
                }
            )
            wandb.log(
                {
                    "expert_2_kernel_lengthscale_2": self.dynamics.mosvgpe.experts_list[
                        1
                    ]
                    .gp.kernel.kernels[0]
                    .lengthscales.numpy()[1]
                }
            )
            wandb.log(
                {
                    "expert_2_kernel_lengthscale_3": self.dynamics.mosvgpe.experts_list[
                        1
                    ]
                    .gp.kernel.kernels[0]
                    .lengthscales.numpy()[2]
                }
            )
            wandb.log(
                {
                    "expert_2_kernel_lengthscale_4": self.dynamics.mosvgpe.experts_list[
                        1
                    ]
                    .gp.kernel.kernels[0]
                    .lengthscales.numpy()[3]
                }
            )
