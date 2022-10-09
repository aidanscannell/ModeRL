#!/usr/bin/env python3
from .multiclass import SoftmaxSerializable
from .scalar_continuous import GaussianSerializable
from .scalar_discrete import BernoulliSerializable


LIKELIHOODS = [GaussianSerializable, BernoulliSerializable, SoftmaxSerializable]
LIKELIHOOD_OBJECTS = {likelihood.__name__: likelihood for likelihood in LIKELIHOODS}
