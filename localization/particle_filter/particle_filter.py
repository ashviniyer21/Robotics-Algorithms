"""
Python file to run and visualize particle filter algorithm
Assumes point mass motion model and range-bearing observation model
Assumption landmarks are in order, evaluation will simply be squared sum of distances for each landmark
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from localization.simulation.motion_model import PointMass2DMotionModel
from localization.simulation.observation_model import RangeBearingModel
import copy

def get_random_points(num, bounds):
    """Generates a random point within bounds"""
    x_points = np.random.uniform(bounds[0][0], bounds[0][1], size=num)
    y_points = np.random.uniform(bounds[1][0], bounds[1][1], size=num)
    theta_points = np.random.uniform(bounds[2][0], bounds[2][1], size=num)
    return np.array([x_points, y_points, theta_points]).T

def motion_model(prev_state:np.array, control:np.array, dt:float):
    """Motion model for moving points"""
    new_state = np.zeros(3)
    new_state[0] = prev_state[0] + control[0] * np.cos(prev_state[2]) * dt
    new_state[1] = prev_state[1] + control[0] * np.sin(prev_state[2]) * dt
    new_state[2] = prev_state[2] + control[1] * dt
    return new_state

def update_point_motion(points:np.array, ctrl:np.array, var:np.array, dt:float):
    """Updates all points with some slightly noise motion contorl"""
    for i in range(len(points)):
        noisy_ctrl = np.random.multivariate_normal(ctrl, var)
        points[i,:] = motion_model(points[i,:], noisy_ctrl, dt)
    return points

def predict_landmarks(point:np.array, observations:np.array):
    """Predicts location of each landmark given observations and a particle"""
    predicted_landmarks = []
    for observation in observations:
        dist = observation[0]
        angle = observation[1]
        combined_angle = angle + point[2]
        x_offset = dist * np.cos(combined_angle)
        y_offset = dist * np.sin(combined_angle)
        predicted_landmark = [point[0] + x_offset, point[1] + y_offset]
        predicted_landmarks.append(predicted_landmark)
    return np.array(predicted_landmarks)

def evaluate_point(predicted_landmarks:np.array, landmarks:np.array):
    """Evaluates how good a given point is based on observations provided"""
    sum_dist = 0
    for i in range(len(predicted_landmarks)):
        sum_dist += np.sqrt((predicted_landmarks[i,0] - landmarks[i,0])**2 + (predicted_landmarks[i,1] - landmarks[i,1])**2)
    return 1 / sum_dist

def calculate_weights(points:np.array, observations:np.array, landmarks:np.array):
    """Calculates weights for each point"""
    weights = []
    weight_sum = 0
    for point in points:
        predicted_landmarks = predict_landmarks(point, observations)
        evaluation = evaluate_point(predicted_landmarks, landmarks)
        weights.append(evaluation)
        weight_sum += evaluation
    return np.array(weights) / weight_sum

def resample(points:np.array, weights:np.array):
    """Resamples points"""
    return points[np.random.choice(np.arange(len(points)), size=len(points), p=weights),:]

def replace(points:np.array, num_replace:int, bounds):
    points_to_replace = np.random.choice(np.arange(len(points)), size=num_replace)
    new_points = get_random_points(num_replace, bounds)
    points[points_to_replace,:] = new_points
    return points

def particle_filter_step(points, ctrl, ctrl_var, observations, landmarks, dt, bounds, num_replace=0):
    """Runs 1 step of the particle filter"""
    weights = calculate_weights(points, observations, landmarks)
    points = resample(points, weights)
    points = replace(points, num_replace, bounds)
    points = update_point_motion(points, ctrl, ctrl_var, dt)
    return points

def plot(state, particles, landmarks):
    """Plots RRT* tree"""
    g = nx.Graph()
    for i in range(len(particles)+1):
        g.add_node(i)
    pos = copy.deepcopy(particles)
    state = np.reshape(state, (1, 3))
    pos = np.concatenate((pos, state), axis=0)
    color_map = ["blue"] * len(pos)
    color_map[len(pos) - 1] = "orange"

    fig, ax = plt.subplots()

    for landmark in landmarks:
        circle=plt.Circle((landmark[0],landmark[1]),1)
        ax.add_patch(circle)

    nx.draw(g, pos=pos, node_color = color_map, node_size=10)
    plt.xlim(bounds[0][0], bounds[0][1])
    plt.ylim(bounds[1][0], bounds[1][1])
    plt.show()

if __name__ == "__main__":
    true_state = np.array([0, 0, 0])
    lin_var = np.array([
        [1, 0],
        [0, 0]
    ]) * 0.01
    ang_var = np.array([
        [0, 0],
        [0, 0.1]
    ]) * 0.01
    const_var = np.array([
        [1, 0],
        [0, 0.1]
    ]) * 0.01

    obs_var = np.array([
        [1, 0],
        [0, 0.1]
    ]) * 0.01

    landmarks = np.random.uniform(-10, 10, (20, 2))

    mot_model = PointMass2DMotionModel(lin_var, ang_var, const_var)
    obs_model = RangeBearingModel(obs_var)

    DT = 0.05

    time = 0

    END_TIME = 100

    bounds=[[-10, 10], [-10, 10], [-np.pi, np.pi]]

    particles = get_random_points(1000, bounds)

    ctrl_var = np.array([
        [1, 0],
        [0, 0.1]
    ])*0.1


    while time < END_TIME:
        control = [1, 0.2]

        true_state = mot_model.sample_model(true_state, control, DT)
        observations = obs_model.sample_observations(true_state, landmarks)

        particles = particle_filter_step(particles, control, ctrl_var, observations, landmarks, DT, bounds, num_replace=10)

        plot(true_state, particles, landmarks)
        
        time += DT