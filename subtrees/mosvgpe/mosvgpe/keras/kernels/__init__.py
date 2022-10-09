#!/usr/bin/env python3
from .multioutput import SeparateIndependentSerializable, SharedIndependentSerializable
from .stationaries import (
    Matern12Serializable,
    Matern32Serializable,
    Matern52Serializable,
    RBFSerializable,
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
