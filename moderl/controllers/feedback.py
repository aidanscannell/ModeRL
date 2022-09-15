#!/usr/bin/env python3
import abc

from moderl.custom_types import State

from .base import FeedbackController


class Policy(FeedbackController):
    @abc.abstractmethod
    def __call__(self, state: State, timestep: int):
        raise NotImplementedError
