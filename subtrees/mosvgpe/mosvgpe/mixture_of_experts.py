#!/usr/bin/env python3
import abc
from typing import List, Optional

import tensor_annotations.tensorflow as ttf
import tensorflow as tf
import tensorflow_probability as tfp
from gpflow import default_float
from gpflow.models import BayesianModel

from .custom_types import Dataset, DatasetBatch, InputData
from .experts import EXPERT_OBJECTS, ExpertBase, SVGPExpert
from .gating_networks import (
    GATING_NETWORK_OBJECTS,
    GatingNetworkBase,
    SVGPGatingNetwork,
)

tf.keras.backend.set_floatx("float64")
tfd = tfp.distributions


# class MixtureOfExpertsBase(gpf.models.BayesianModel, abc.ABC):
# class MixtureOfExpertsBase(tf.keras.Model, abc.ABC):
class MixtureOfExpertsBase(tf.keras.Model, BayesianModel, abc.ABC):
    r"""Interface for mixture of experts models.

    Given an input :math:`x` and an output :math:`y` the mixture of experts
    marginal likelihood is given by,

    .. math::
        p(y|x) = \sum_{k=1}^K \Pr(\\alpha=k | x) p(y | \\alpha=k, x)

    Assuming the expert indicator variable :math:`\\alpha \in \{1, ...,K\}`
    the mixing probabilities are given by :math:`\Pr(\\alpha=k | x)` and are
    collectively referred to as the gating network.
    The experts are given by :math:`p(y | \\alpha=k, x)` and are responsible for
    predicting in different regions of the input space.

    Each subclass that inherits MixtureOfExperts should implement the
    maximum_log_likelihood_objective(data) method. It is used as the objective
    function to optimise the models trainable parameters.
    """

    def __init__(
        self,
        experts_list: List[ExpertBase],
        gating_network: GatingNetworkBase,
        name: str = "MixtureOfExperts",
    ):
        super().__init__(name=name)
        self._experts_list = experts_list
        self._gating_network = gating_network

        for expert in experts_list:
            assert isinstance(expert, ExpertBase)
        assert isinstance(gating_network, GatingNetworkBase)
        assert gating_network.num_experts == self.num_experts

    def call(self, Xnew: InputData, training: Optional[bool] = True):
        if training:
            return self.predict_y(Xnew)
        else:
            raise NotImplementedError(
                "mixture of experts model should implement functionality for training=True"
            )

    def predict_y(self, Xnew: InputData, **kwargs) -> tfd.Mixture:
        # TODO should there be separate kwargs for gating and experts?
        """Predicts the mixture distribution at Xnew.

        Mixture dist has
        batch_shape == [NumData]
        event_shape == [OutputDim] # TODO is this true for multi output setting and full_cov?
        """
        expert_indicator_categorical_dist = self.gating_network(Xnew, **kwargs)
        experts_dists = self.predict_experts_dists(Xnew, **kwargs)
        return tfd.Mixture(
            cat=expert_indicator_categorical_dist, components=experts_dists
        )

    def predict_experts_dists(
        self, Xnew: InputData, **kwargs
    ) -> List[tfd.Distribution]:
        """Calculates each experts predictive distribution at Xnew.

        Each expert's dist has shape [NumData, OutputDim, NumExperts]
        if not full_cov:
            batch_shape == [[NumData], [NumData], ...] with len(batch_shape)=NumExperts
            event_shape == [[OutputDim], [OutputDim], ...] with len(event_shape)=NumExperts
        """
        return [expert(Xnew, **kwargs) for expert in self.experts_list]

    @property
    def experts_list(self) -> List[ExpertBase]:
        return self._experts_list

    @property
    def gating_network(self) -> GatingNetworkBase:
        return self._gating_network

    @property
    def num_experts(self) -> int:
        return len(self.experts_list)

    def get_config(self):
        experts_list = []
        for expert in self.experts_list:
            experts_list.append(tf.keras.layers.serialize(expert))
        return {
            "experts_list": experts_list,
            "gating_network": tf.keras.layers.serialize(self.gating_network),
        }

    @classmethod
    def from_config(cls, cfg: dict):
        expert_list = []
        for expert_cfg in cfg["experts_list"]:
            expert_list.append(
                tf.keras.layers.deserialize(expert_cfg, custom_objects=EXPERT_OBJECTS)
            )
        gating_network = tf.keras.layers.deserialize(
            cfg["gating_network"], custom_objects=GATING_NETWORK_OBJECTS
        )
        return cls(experts_list=expert_list, gating_network=gating_network)


class MixtureOfSVGPExperts(MixtureOfExpertsBase):
    """Mixture of SVGP experts using stochastic variational inference.

    Implemention of a Mixture of Gaussian Process Experts method where both
    the gating network and experts are modelled as SVGPs.
    The model is trained with stochastic variational inference by exploiting
    the factorization achieved by sparse GPs.
    """

    def __init__(
        self,
        experts_list: List[SVGPExpert],
        gating_network: SVGPGatingNetwork,
        num_samples: Optional[int] = 1,
        num_data: Optional[int] = None,
        name: str = "MoSVGPE",
    ):
        for expert in experts_list:
            assert isinstance(expert, SVGPExpert)
        assert isinstance(gating_network, SVGPGatingNetwork)
        super().__init__(
            experts_list=experts_list, gating_network=gating_network, name=name
        )
        self.num_data = num_data
        self.num_samples = num_samples
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        # self.gating_kl_tracker = tf.keras.metrics.Mean(name="gating_kl")
        # self.experts_kl_tracker = tf.keras.metrics.Mean(name="expert_kl")

    def call(
        self, Xnew: InputData, training: Optional[bool] = False
    ) -> tfd.MixtureSameFamily:
        if not training:
            # return self.predict_y(Xnew)
            dist = self.predict_y(Xnew)
            return dist.mean(), dist.variance()

    def train_step(self, data: DatasetBatch):
        with tf.GradientTape() as tape:
            loss = -self.maximum_log_likelihood_objective(data)

        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        self.loss_tracker.update_state(loss)
        return {
            "loss": self.loss_tracker.result(),
            # "gating_kl": self.gating_kl_tracker.result(),
            # "experts_kl": self.experts_kl_tracker.result(),
        }

    def test_step(self, data):
        loss = -self.maximum_log_likelihood_objective(data)
        return {"loss": loss}

    def maximum_log_likelihood_objective(self, data: Dataset) -> ttf.Tensor0:
        return self.elbo(
            data=data, num_samples=self.num_samples, num_data=self.num_data
        )

    def elbo(
        self,
        data: DatasetBatch,
        num_samples: Optional[int] = 1,
        num_data: Optional[int] = None,
    ) -> ttf.Tensor0:
        r"""Lower bound to the log-marginal likelihood (ELBO).

        This bound removes the M dimensional integral over the gating
        network inducing variables $q(\hat{\mathbf{U}})$ with 1 dimensional
        integrals over the gating network variational posterior $q(\mathbf{h}_n)$.
        """
        X, Y = data

        kl_gating = tf.reduce_sum(self.gating_network.prior_kl())
        kl_experts = tf.reduce_sum([expert.prior_kl() for expert in self.experts_list])

        # Evaluate gating network to get samples of categorical dist over inicator var
        mixing_probs = self.gating_network.predict_categorical_dist(
            X, num_samples=num_samples
        ).probs  # [S, N, K]
        print("Mixing probs: {}".format(mixing_probs.shape))

        # Evaluate experts
        Y = tf.expand_dims(Y, 0)  # [S, N, F]
        # print("Y: {}".format(Y.shape))
        experts_probs = [
            expert.predict_dist_given_inducing_samples(X, num_samples).prob(Y)
            for expert in self.experts_list
        ]
        experts_probs = tf.stack(experts_probs, -1)  # [S, N, K]
        print("Experts probs: {}".format(experts_probs.shape))

        tf.debugging.assert_shapes(
            [
                (experts_probs, ["S1", "N", "K"]),
                (mixing_probs, ["S2", "N", "K"]),
            ],
            message="Gating network and experts dimensions do not match",
        )

        # Expand to enable integrationg over both expert and gating samples
        experts_probs = experts_probs[:, tf.newaxis, :, tf.newaxis, :]
        mixing_probs = mixing_probs[tf.newaxis, :, :, :, tf.newaxis]
        # print("Experts probs EXP: {}".format(experts_probs.shape))
        # print("Mixing probs EXP: {}".format(mixing_probs.shape))
        # print("Matmul EXP: {}".format(tf.matmul(experts_probs, mixing_probs).shape))
        marginalised_experts = tf.matmul(experts_probs, mixing_probs)[..., 0, 0]
        # print("Marginalised indicator variable: {}".format(marginalised_experts.shape))

        log_prob = tf.math.log(marginalised_experts)
        var_exp = tf.reduce_mean(log_prob, axis=[0, 1])  # Average gating/expert samples
        var_exp = tf.reduce_sum(var_exp, 0)

        if num_data is not None:
            batch_size = tf.shape(X)[0]
            scale = tf.cast(num_data / batch_size, default_float())
        else:
            scale = tf.cast(1.0, default_float())

        return var_exp * scale - kl_gating - kl_experts
