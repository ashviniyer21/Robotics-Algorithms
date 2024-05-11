"""Class to Store Motion Models"""

from abc import ABC, abstractmethod
import numpy as np

class ObservationModel(ABC):
    """Abstract Class structure for observation models"""

    @abstractmethod
    def sample_observations(self, state, objective):
        """Returns a observation based on state and objective"""

class RangeBearingModel(ObservationModel):
    """Class for Rang Bearing Model"""

    def __init__(self, variance):
        """Initializes class with variances needed"""
        self.variance = variance

    def sample_observations(self, state, landmarks):
        """Samples model for an observation. Samples with noise"""

        observations = np.zeros((len(landmarks), 2))
        for i, landmark in enumerate(landmarks):
            dist_x = landmark[0] - state[0]
            dist_y = landmark[1] - state[1]
            dist = np.sqrt(dist_x**2 + dist_y**2)
            theta = np.arctan2(dist_x, dist_y) - state[2]
            curr_observations = np.random.multivariate_normal([dist, theta], self.variance)
            observations[i, 0] = curr_observations[0]
            observations[i, 1] = curr_observations[1]

        return observations


    def get_variance(self):
        """Returns variance of motion model given control input"""
        return self.variance
    
    def get_total_variance(self, num_landmarks):
        """Returns variance matrix for all landmarks"""
        var = np.zeros((2 * num_landmarks, 2 * num_landmarks))
        for i in range(num_landmarks):
            var[2 * i:2 * i + 2,2 * i:2 * i + 2] = self.variance
        return var
