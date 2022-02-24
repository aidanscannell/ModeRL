#!/usr/bin/env python3
from typing import Optional

import numpy as np
from scipy.optimize import LinearConstraint


def build_linear_control_constraints(
    self,
    horizon: int,
    control_dim: int,
    lower_bound: Optional[float] = None,
    upper_bound: Optional[float] = None,
    gaussian: Optional[
        bool
    ] = False,  # if True adjust constraint matrix for mean AND variance
):
    """Linear constraints on the mean of the control dist."""
    # TODO handle sequence of inputs
    if upper_bound is None and lower_bound is None:
        return None
    if upper_bound is not None:
        constraints_upper_bound = np.ones((horizon, 1)) * upper_bound
    else:
        constraints_upper_bound = np.ones((horizon, 1)) * np.inf
    if lower_bound is not None:
        constraints_lower_bound = np.ones((horizon, 1)) * lower_bound
    else:
        constraints_lower_bound = np.ones((horizon, 1)) * -np.inf
    if gaussian:
        control_constraint_matrix = np.eye(
            N=horizon * control_dim, M=horizon * control_dim * 2
        )
    else:
        control_constraint_matrix = np.eye(horizon * control_dim)
    return LinearConstraint(
        control_constraint_matrix,
        constraints_lower_bound.reshape(-1),
        constraints_upper_bound.reshape(-1),
    )
