#!/usr/bin/env python3
from typing import Union
from .base import FeedbackController, NonFeedbackController
from .non_feedback.trajectory_optimisation import TrajectoryOptimisationController
from .non_feedback.geodesic_collocation import GeodesicController

CONTROLLERS = [TrajectoryOptimisationController, GeodesicController]
CONTROLLER_OBJECTS = {controller.__name__: controller for controller in CONTROLLERS}
