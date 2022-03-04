#!/usr/bin/env python3
from typing import Optional

import numpy as np
import tensorflow_probability as tfp
from modeopt.trajectories import ControlTrajectoryDist
from scipy.optimize import LinearConstraint

tfd = tfp.distributions


def build_linear_control_constraints(
    trajectory: ControlTrajectoryDist, lower_bound=None, upper_bound=None
) -> Optional[LinearConstraint]:
    """Linear constraints on control trajectory"""
    # TODO handle sequence of inputs
    if upper_bound is None and lower_bound is None:
        return None
    if upper_bound is not None:
        constraints_upper_bound = np.ones((trajectory.horizon, 1)) * upper_bound
    else:
        constraints_upper_bound = np.ones((trajectory.horizon, 1)) * np.inf
    if lower_bound is not None:
        constraints_lower_bound = np.ones((trajectory.horizon, 1)) * lower_bound
    else:
        constraints_lower_bound = np.ones((trajectory.horizon, 1)) * -np.inf

    # TODO move this check to dispatcher?
    if isinstance(trajectory.dist, tfd.Deterministic):
        print("building constraints for tfd.Deterministic")
        control_constraint_matrix = np.eye(trajectory.horizon * trajectory.control_dim)
    else:
        control_constraint_matrix = np.eye(
            N=trajectory.horizon * trajectory.control_dim,
            M=trajectory.horizon * trajectory.control_dim * 2,
        )
    return LinearConstraint(
        control_constraint_matrix,
        constraints_lower_bound.reshape(-1),
        constraints_upper_bound.reshape(-1),
    )
