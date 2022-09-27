#!/usr/bin/env python3
from gpflow.kernels import RBF, Matern52, Matern32, Matern12


# class RBFSerializable(RBF, tf.keras.layers.Layer):
class RBFSerializable(RBF):
    def get_config(self):
        if isinstance(self.active_dims, slice):
            # TODO how to serialise slice?
            cfg = {"active_dims": None}
        else:
            cfg = {"active_dims": self.active_dims}
        cfg.update(
            {
                "lengthscales": self.lengthscales.numpy(),
                "variance": self.variance.numpy(),
            }
        )
        return cfg

    @classmethod
    def from_config(cls, cfg: dict):
        return cls(**cfg)


class Matern52Serializable(Matern52):
    def get_config(self):
        if isinstance(self.active_dims, slice):
            # TODO how to serialise slice?
            cfg = {"active_dims": None}
        else:
            cfg = {"active_dims": self.active_dims}
        cfg.update(
            {
                "lengthscales": self.lengthscales.numpy(),
                "variance": self.variance.numpy(),
            }
        )
        return cfg

    @classmethod
    def from_config(cls, cfg: dict):
        return cls(**cfg)


class Matern32Serializable(Matern52):
    def get_config(self):
        if isinstance(self.active_dims, slice):
            # TODO how to serialise slice?
            cfg = {"active_dims": None}
        else:
            cfg = {"active_dims": self.active_dims}
        cfg.update(
            {
                "lengthscales": self.lengthscales.numpy(),
                "variance": self.variance.numpy(),
            }
        )
        return cfg

    @classmethod
    def from_config(cls, cfg: dict):
        return cls(**cfg)


class Matern12Serializable(Matern52):
    def get_config(self):
        if isinstance(self.active_dims, slice):
            # TODO how to serialise slice?
            cfg = {"active_dims": None}
        else:
            cfg = {"active_dims": self.active_dims}
        cfg.update(
            {
                "lengthscales": self.lengthscales.numpy(),
                "variance": self.variance.numpy(),
            }
        )
        return cfg

    @classmethod
    def from_config(cls, cfg: dict):
        return cls(**cfg)
