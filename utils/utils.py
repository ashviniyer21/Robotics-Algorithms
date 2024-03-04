"""File that contains useful general math functions"""

import numpy as np

def wrap_angle(angle):
    """Returns the angle clipped between (-pi, pi)"""

    new_angle  = angle % (2 * np.pi)
    if new_angle > np.pi:
        new_angle -= 2 * np.pi
    return new_angle
