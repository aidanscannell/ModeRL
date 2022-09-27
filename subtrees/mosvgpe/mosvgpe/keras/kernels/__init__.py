#!/usr/bin/env python3
from .multioutput import SeparateIndependentSerializable, SharedIndependentSerializable
from .stationaries import (
    RBFSerializable,
    Matern12Serializable,
    Matern32Serializable,
    Matern52Serializable,
)

KERNELS = [
    RBFSerializable,
    SharedIndependentSerializable,
    SeparateIndependentSerializable,
    Matern12Serializable,
    Matern32Serializable,
    Matern52Serializable,
]
KERNEL_OBJECTS = {kernel.__name__: kernel for kernel in KERNELS}
