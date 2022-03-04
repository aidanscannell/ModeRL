#!/usr/bin/env python3
from .base import BaseTrajectory
from .geodesics import GeodesicTrajectory
from .variational import (
    ControlTrajectoryDist,
    initialise_deterministic_trajectory,
    initialise_gaussian_trajectory,
)

TRAJECTORIES = [GeodesicTrajectory, ControlTrajectoryDist]
TRAJECTORY_OBJECTS = {traj.__name__: traj for traj in TRAJECTORIES}
