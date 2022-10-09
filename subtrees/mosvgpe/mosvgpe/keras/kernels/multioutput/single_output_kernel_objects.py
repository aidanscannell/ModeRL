#!/usr/bin/env python3
from ..stationaries import RBFSerializable


SINGLE_OUTPUT_KERNELS = [RBFSerializable]
SINGLE_OUTPUT_KERNEL_OBJECTS = {
    kernel.__name__: kernel for kernel in SINGLE_OUTPUT_KERNELS
}
