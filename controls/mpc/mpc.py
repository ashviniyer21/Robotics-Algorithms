"""Python file to run and visualize MPC Controller"""

import gymnasium as gym
import numpy as np

def extrapolate_pos(state, lens=[0.1, 0.1]):
    angle0 = state[0,0]
    angle1 = state[1,0]
    x = lens[0] * np.cos(angle0) + lens[1] * np.cos(angle0 + angle1)
    y = lens[0] * np.sin(angle0) + lens[1] * np.sin(angle0 + angle1)
    return np.array([x, y])

def calculate_state(init_state, torques, dt, mass=0.05, lens=[0.1, 0.1]):
    horzion_length = torques.shape[0]
    states = np.zeros((horzion_length, 2, 2))
    states[0] = init_state
    for i in range(1, horzion_length):
        prev_state = states[i - 1]
        pos = prev_state[:,0]
        vel = prev_state[:,1]
        acc = torques[i,:] / lens
        acc /= mass
        new_pos = pos + vel * dt
        new_vel = vel + acc * dt
        states[i,:,0] = new_pos
        states[i,:,1] = new_vel
    return states
def calculate_cost(state, torques, target, gamma=0.98):
    horizon_length = state.shape[0]
    cost = 0
    for i in range(horizon_length):
        pos = extrapolate_pos(state[i])
        vel = state[i,:,1]
        torque = torques[i]
        cost *= gamma
        dist = np.linalg.norm(target - pos)
        cost += dist + 0.5 * np.linalg.norm(torque)**2
    return cost


def mpc(state, target, dt, horizon_length=5, rollouts=100, top_rollouts=10, iters=5):
    torques = np.random.uniform(-1, 1, (rollouts, horizon_length, 2))
    costs = np.zeros((rollouts))
    for i in range(iters):
        for j in range(rollouts):
            states = calculate_state(state, torques[j], dt)
            costs[j] = calculate_cost(states, torques[j], target)
        idxs = np.argpartition(costs, top_rollouts)
        top_torques = torques[idxs[:top_rollouts]]
        top_torque = torques[np.argmin(costs),0,:]
        avgs = np.mean(top_torques, axis=0)
        std_devs = np.std(top_torques, axis=0)
        if i == iters - 1:
            return top_torque
        torques = np.random.normal(avgs, std_devs, (rollouts, horizon_length, 2))
if __name__ == "__main__":
    DT = 0.02
    GOAL = 0

    env = gym.make('Reacher-v4', render_mode="human", max_episode_steps=500)
    observation, info = env.reset()
    action = [0, 0]
    prev_vel0 = 0
    prev_vel1 = 0
    for _ in range(1000):
        observation, reward, terminated, truncated, info = env.step(action)
        cos0, cos1, sin0, sin1, targetx, targety, vel0, vel1, posx, posy, posz = observation
        # print("Acceleration:", (vel0 - prev_vel0) / DT, (vel1 - prev_vel1) / DT)
        angle0 = np.arctan2(sin0, cos0)
        angle1 = np.arctan2(sin1, cos1)
        state = np.array([[angle0, vel0], [angle1, vel1]])
        target = np.array([targetx, targety])
        action = mpc(state, [targetx, targety], DT)
        # action = env.action_space.sample()
        # print("Torque: ", action)
        prev_vel0 = vel0
        prev_vel1 = vel1
        # action = env.action_space.sample()
        if terminated or truncated:
            observation, info = env.reset()
            if truncated:
                print("fail")
            break

    env.close()
