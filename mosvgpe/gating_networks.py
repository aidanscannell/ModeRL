#!/usr/bin/env python3
from abc import abstractmethod
from typing import Optional, Union

import gpflow as gpf
import tensor_annotations.tensorflow as ttf
import tensorflow as tf
import tensorflow_probability as tfp
from gpflow.inducing_variables import InducingVariables
from gpflow.kernels import Kernel, MultioutputKernel
from gpflow.mean_functions import MeanFunction
from gpflow.models.model import GPModel

from .custom_types import (
    ExpertIndicatorCategoricalDist,
    GatingFunctionSamples,
    GatingMeanAndVariance,
    InputData,
    MixingProb,
    MixingProbSamples,
    NumData,
    NumGatingFunctions,
    NumSamples,
)
from .gp import predict_f_given_inducing_samples

tfd = tfp.distributions


class GatingNetworkBase(gpf.Module, tf.keras.layers.Layer):
    # class GatingNetworkBase(gpf.Module):
    """Interface for gating networks"""

    def __init__(self, num_experts: int, name: Optional[str] = "gating_network"):
        super().__init__(name=name)
        self._num_experts = num_experts

    def predict_categorical_dist(self, Xnew: InputData, **kwargs) -> tfd.Categorical:
        r"""Return probability mass function over expert indicator as Categorical dist

        .. math::
            P(\alpha | Xnew)
        batch_shape == NumData
        """
        mixing_probs = self.predict_mixing_probs(Xnew, **kwargs)
        return tfd.Categorical(probs=mixing_probs, name="ExpertIndicatorCategorical")

    @abstractmethod
    def predict_mixing_probs(self, Xnew: InputData, **kwargs) -> MixingProb:
        r"""Calculates the set of experts mixing probabilities at Xnew

        :math:`\{\Pr(\\alpha=k | x)\}^K_{k=1}`
        """
        raise NotImplementedError

    @property
    def num_experts(self):
        return self._num_experts


class GPGatingNetworkBase(GatingNetworkBase):
    """Interface for GP-based gating network"""

    def __init__(self, gp: GPModel, num_experts: int, name: str = "gp_gating_network"):
        super().__init__(num_experts=num_experts, name=name)
        self._gp = gp

    @abstractmethod
    def predict_mixing_probs(self, Xnew: InputData, **kwargs) -> MixingProb:
        r"""Calculates the set of experts mixing probabilities at Xnew

        :math:`\{\Pr(\\alpha=k | x)\}^K_{k=1}`
        """
        raise NotImplementedError

    def predict_f(
        self, Xnew: InputData, full_cov: Optional[bool] = True
    ) -> GatingMeanAndVariance:
        """Calculates the set of gating function GP posteriors at Xnew"""
        return self.gp.predict_f(Xnew, full_cov=full_cov)

    @property
    def gp(self):
        return self._gp


class SVGPGatingNetwork(GPGatingNetworkBase):
    def __init__(
        self,
        kernel: Kernel,
        inducing_variable: InducingVariables,
        mean_function: MeanFunction = None,
        q_diag: bool = False,
        q_mu=None,
        q_sqrt=None,
        whiten: bool = True,
        name: Optional[str] = "SVGPGatingNetwork",
    ):
        if isinstance(kernel, MultioutputKernel):
            self.num_gating_gps = kernel.num_latent_gps
            num_experts = self.num_gating_gps
            likelihood = gpf.likelihoods.Softmax(num_classes=self.num_gating_gps)
        else:
            self.num_gating_gps = 1
            num_experts = 2
            likelihood = gpf.likelihoods.Bernoulli()

        svgp = gpf.models.SVGP(
            kernel=kernel,
            likelihood=likelihood,
            inducing_variable=inducing_variable,
            mean_function=mean_function,
            num_latent_gps=self.num_gating_gps,
            q_diag=q_diag,
            q_mu=q_mu,
            q_sqrt=q_sqrt,
            whiten=whiten,
            num_data=None,
        )

        super().__init__(gp=svgp, num_experts=num_experts, name=name)

    def call(
        self,
        Xnew: InputData,
        num_samples: Optional[int] = 1,
        training: Optional[bool] = False,
    ) -> Union[MixingProbSamples, ExpertIndicatorCategoricalDist]:
        # TODO move this to training
        if training:
            return self.predict_categorical_dist(
                Xnew, num_samples=num_samples
            )  # [S, N, K]
        else:
            return self.predict_categorical_dist(Xnew)

    def predict_categorical_dist(
        self, Xnew: InputData, num_samples: Optional[int] = None
    ) -> tfd.Categorical:
        r"""Return probability mass function over expert indicator as Categorical dist

        :return: categorical dist with batch_shape [NumData] or [NumSamples, NumData]
        """
        mixing_probs = self.predict_mixing_probs(Xnew, num_samples=num_samples)
        return tfd.Categorical(probs=mixing_probs, name="ExpertIndicatorCategorical")

    def predict_mixing_probs(
        self,
        Xnew: InputData,
        num_samples: Optional[int] = None,
        full_cov: Optional[bool] = False,
    ) -> Union[MixingProb, MixingProbSamples]:
        r"""Compute mixing probabilities at Xnew

        Pr(\alpha = k | Xnew) = \int Pr(\alpha = k | h) p(h | Xnew) dh
        """
        if num_samples:
            h_samples = self.predict_h_samples(
                Xnew, num_samples=num_samples, full_cov=full_cov
            )  # [S, N, K]
            probs = self.gp.likelihood.conditional_mean(Xnew, h_samples)  # [S, N, K]
        else:
            h_mean, h_var = self.predict_h(Xnew, full_cov=full_cov)
            probs = self.gp.likelihood.predict_mean_and_var(Xnew, h_mean, h_var)[
                0
            ]  # [N, K]
        return probs

    def predict_h(
        self, Xnew: InputData, full_cov: Optional[bool] = False
    ) -> GatingMeanAndVariance:
        h_mean, h_var = self.gp.predict_f(Xnew, full_cov=full_cov)
        if self.num_gating_gps == 1:
            h_mean = tf.concat([h_mean, -h_mean], -1)
            if full_cov:
                h_var = tf.concat([h_var, h_var], 0)
            else:
                h_var = tf.concat([h_var, h_var], -1)
        return h_mean, h_var

    def predict_h_samples(
        self,
        Xnew: InputData,
        num_samples: Optional[int] = None,
        full_cov: Optional[bool] = False,
    ) -> GatingFunctionSamples:
        h_samples = self.gp.predict_f_samples(
            Xnew, num_samples=num_samples, full_cov=full_cov
        )
        if self.num_gating_gps == 1:
            h_samples = tf.concat([h_samples, -h_samples], -1)
        return h_samples

    def prior_kl(self) -> ttf.Tensor1[NumGatingFunctions]:
        """Returns the gating functions' KL divergence(s)"""
        return self.gp.prior_kl()

    @property
    def gp(self):
        return self._gp
