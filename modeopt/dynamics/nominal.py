#!/usr/bin/env python3


def velocity_controlled_point_mass_dynamics(
    state_mean, control_mean, state_var=None, control_var=None, delta_time=0.05
):
    velocity = control_mean
    delta_state_mean = velocity * delta_time
    delta_state_var = control_var * delta_time ** 2
    return delta_state_mean, delta_state_var
