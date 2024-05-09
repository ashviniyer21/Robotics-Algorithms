"""Python file to run and visualize LQR Controller"""

import gymnasium as gym
import numpy as np

def extrapolate_angle(goal, lens=[0.1, 0.1]):
    """Converts x / y goal to angles using 2-link IK"""
    cos_angle1 = (goal[0] ** 2 + goal[1] ** 2 - lens[0] ** 2 - lens[1] ** 2) / (2 * lens[0] * lens[1])
    angle1 = np.arccos(cos_angle1)
    angle0 = np.arctan2(goal[1], goal[0]) - np.arctan2(lens[1] * np.sin(angle1), lens[0] + lens[1] * np.cos(angle1))
    return np.array([angle0, angle1])

def lqr(angles, vel, goal, dt, Q, R, iters=100, lens=np.array([0.1, 0.1]), mass=np.array([0.05, 0.05])):
    """LQR Controller for 2-link arm (targets new position w/ zero velocity w/ torque control)"""
    angle_goal = extrapolate_angle(goal)
    state_goal = np.array([angle_goal[0], angle_goal[1], 0, 0])
    state = np.array([angles[0], angles[1], vel[0], vel[1]])
    error = state - state_goal

    #Finding LQR state / control matrices
    A = np.array([
        [1, 0, dt, 0],
        [0, 1, 0, dt],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
        ])
    #Moment of inertia
    I = 1/3 * mass * lens * lens

    #v = v + a * dt
    #T = I * a
    #v = v + T/I * dt
    B = np.array([
        [0, 0],
        [0, 0],
        [1/I[0] * dt, 0],
        [0, 1/I[1] * dt]
    ])

    # Computing LQR formula using Dynamic Programming
    P = Q
    for _ in range(iters):
        P = Q + A.T @ P @ A - (A.T @ P @ B) @ np.linalg.pinv(
            R + B.T @ P @ B) @ (B.T @ P @ A)
    
    K = -np.linalg.pinv(R + B.T @ P @ B) @ B.T @ P @ A
    u = K @ error
    return u


if __name__ == "__main__":
    DT = 0.01
    GOAL = 0

    env = gym.make('Reacher-v4', render_mode="human", max_episode_steps=250)
    observation, info = env.reset()
    action = [0, 0]
    for _ in range(1000):
        observation, reward, terminated, truncated, info = env.step(action)
        cos0, cos1, sin0, sin1, targetx, targety, vel0, vel1, posx, posy, posz = observation
        angle0 = np.arctan2(sin0, cos0)
        angle1 = np.arctan2(sin1, cos1)
        angles = np.array([angle0, angle1])
        vels = np.array([vel0, vel1])
        target = np.array([targetx, targety])
        Q = np.array([
            [10, 0, 0, 0],
            [0, 10, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        R = np.array([
            [0.1, 0],
            [0, 0.1]
        ])
        action = lqr(angles, vels, target, DT, Q, R)
        if terminated or truncated:
            observation, info = env.reset()
            break

    env.close()
