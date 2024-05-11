"""Class to Store Motion Models"""

from abc import ABC, abstractmethod
import numpy as np

class MotionModel(ABC):
    """Abstract Class structure for motion models"""

    @abstractmethod
    def sample_model(self, state, control, dt):
        """Returns a new state based on control input and noise"""

class PointMass2DMotionModel(MotionModel):
    """Class for Point Mass 2D Motion Model"""

    def __init__(self, variance_lin, variance_ang, variance_const):
        """Initializes class with variances needed"""
        self.variance_lin = variance_lin
        self.variance_ang = variance_ang
        self.variance_const = variance_const

    def sample_model(self, state, control, dt):
        """Samples model for new state given control input. Samples with noise"""

        control_var = self.get_variance(control)
        control = np.random.multivariate_normal(control, control_var)

        new_state = np.zeros(3)
        new_state[0] = state[0] + control[0] * np.cos(state[2]) * dt
        new_state[1] = state[1] + control[0] * np.sin(state[2]) * dt
        new_state[2] = state[2] + control[1] * dt
        return new_state

    def get_variance(self, control):
        """Returns variance of motion model given control input"""
        return self.variance_lin * control[0] + self.variance_ang * control[1] + self.variance_const

    def get_state_variance(self, control, dt):
        """Returns variance in terms of state"""
        control_var = self.get_variance(control)
        state_var = np.zeros((3, 3))
        state_var[0] = [control_var[0, 0], control_var[0, 0], control_var[0, 1]]
        state_var[1] = [control_var[0, 0], control_var[0, 0], control_var[0, 1]]
        state_var[2] = [control_var[1, 0], control_var[1, 0], control_var[1, 1]]
        state_var *= dt
        return state_var
