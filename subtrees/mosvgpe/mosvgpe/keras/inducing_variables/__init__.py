#!/usr/bin/env python3
from .inducing_variables import InducingPointsSerializable

# from .multioutput.inducing_variables import Shared
from .multioutput import (
    SharedIndependentInducingVariablesSerializable,
    SeparateIndependentInducingVariablesSerializable,
)

INDUCING_VARIABLES = [
    InducingPointsSerializable,
    SharedIndependentInducingVariablesSerializable,
    SeparateIndependentInducingVariablesSerializable,
]
INDUCING_VARIABLE_OBJECTS = {
    inducing_variable.__name__: inducing_variable
    for inducing_variable in INDUCING_VARIABLES
}
