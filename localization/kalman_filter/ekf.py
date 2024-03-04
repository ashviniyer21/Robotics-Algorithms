"""Python file to run and visualize EKF algorithm for simple robot
   Robot has state (x, y, theta) and control (v, omega)
   Robot also has observed landmarks it can always see
   Obseveration Model is (range, theta) distance between robot and landmark"""

import numpy as np
from localization.simulation.motion_model import PointMass2DMotionModel
from localization.simulation.observation_model import RangeBearingModel
import utils.utils as utils

def motion_update(prev_state, control, dt):
    """Motion Model For point mass robot"""

    new_state = np.zeros(3)
    new_state[0] = prev_state[0] + control[0] * np.cos(prev_state[2]) * dt
    new_state[1] = prev_state[1] + control[0] * np.sin(prev_state[2]) * dt
    new_state[2] = prev_state[2] + control[1] * dt

    return new_state

def motion_jacobian(prev_state, control, dt):
    """Jacobian of Motion Model"""

    jacobian = np.zeros((3, 3))
    jacobian[0] = [1, 0, -control[0] * np.sin(prev_state[2]) * dt]
    jacobian[1] = [0, 1, control[0] * np.cos(prev_state[2]) * dt]
    jacobian[2] = [0, 0, 1]

    return jacobian

def observation_model(state, landmarks):
    """Observation Model for Range Bearing Sensor"""

    observations = np.zeros(2 * len(landmarks))

    for i, landmark in enumerate(landmarks):
        dist_x = landmark[0] - state[0]
        dist_y = landmark[1] - state[1]
        dist = np.sqrt(dist_x**2 + dist_y**2)
        theta = np.arctan2(dist_x, dist_y) - state[2]
        observations[2 * i] = dist
        observations[2 * i + 1] = theta

    return observations

def observation_jacobian(state, landmarks):
    """Jacobian of Observation Model"""

    jacobian = np.zeros((2 * len(landmarks), 3))
    for i, landmark in enumerate(landmarks):
        landmark_x = landmark[0] - state[0]
        landmark_y = landmark[1] - state[1]
        jacobian[2 * i, 0] = -landmark_x / np.sqrt(landmark_x**2 + landmark_y**2)
        jacobian[2 * i, 1] = -landmark_y / np.sqrt(landmark_x**2 + landmark_y**2)
        jacobian[2 * i, 2] = 0
        jacobian[2 * i + 1, 0] = landmark_y / ((landmark_y**2 / landmark_x**2 + 1) * landmark_x**2)
        jacobian[2 * i + 1, 1] = -1 / ((landmark_y**2 / landmark_x**2 + 1) * landmark_x)
        jacobian[2 * i + 1, 2] = -1
    return jacobian

def ekf_update(prev_state, prev_variance, control, observation, landmarks, motion_variance,
               observation_variance, dt):
    """Updating motion prediction at new time step using Extended Kalman Filtering"""
    new_state = motion_update(prev_state, control, dt)
    motion_j = motion_jacobian(prev_state, control, dt)
    new_variance = motion_j @ prev_variance @ motion_j.T + motion_variance

    observation_j = observation_jacobian(new_state, landmarks)
    kalman_gain = new_variance @ observation_j.T @ np.linalg.inv(
        observation_j @ new_variance @ observation_j.T + observation_variance)
    new_state = new_state + kalman_gain @ (observation - observation_model(new_state, landmarks))
    new_variance = (np.identity(len(new_state)) - kalman_gain @ observation_j) @ new_variance

    return new_state, new_variance

if __name__ == "__main__":
    start_state = np.array([0, 0, 0])
    lin_var = np.array([
        [1, 0],
        [0, 0]
    ]) * 0.05
    ang_var = np.array([
        [0, 0],
        [0, 1]
    ]) * 0.05
    const_var = np.array([
        [1, 0],
        [0, 1]
    ]) * 0.05

    obs_var = np.array([
        [1, 0],
        [0, 1]
    ]) * 0.05

    landmarks = np.random.uniform(-10, 10, (10, 2))

    mot_model = PointMass2DMotionModel(lin_var, ang_var, const_var)
    obs_model = RangeBearingModel(obs_var)

    DT = 0.02

    time = 0

    END_TIME = 100

    state = np.zeros(3)
    pred_state = np.zeros(3)
    pred_var = np.zeros((3, 3))
    while time < END_TIME:
        control = [np.random.uniform(-0.5, 0.5), np.random.uniform(-0.2, 0.2)]

        state = mot_model.sample_model(state, control, DT)
        observation = obs_model.sample_observations(state, landmarks)
        motion_var = mot_model.get_state_variance(control, DT)
        total_obs_var = obs_model.get_total_variance(len(landmarks))

        pred_state, pred_var = ekf_update(pred_state, pred_var, control, observation, landmarks,
                                          motion_var, total_obs_var, DT)

        time += DT
