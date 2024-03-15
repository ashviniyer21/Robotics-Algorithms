"""Python file to run and visualize PID Controller"""

import gymnasium as gym

class PIDController:
    """Class to run PID controller"""

    def __init__(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.prev_error = None
        self.total_error = 0

    def update_step(self, state, setpoint, dt):
        """Returns new control output given next iteration of PID Controller"""
        error = setpoint - state
        self.total_error += error * dt
        control = 0
        if self.prev_error is None:
            control = self.kp * error
        else:
            control = self.kp * error + self.ki * self.total_error + (
                 self.kd * (error - self.prev_error) / dt)

        self.prev_error = state
        return control

    def integrator_reset(self):
        """Resets integrator, useful when errors become too large"""
        self.total_error = 0

if __name__ == "__main__":
    pid_controller = PIDController(20, 0.02, 2)
    DT = 0.02
    GOAL = 0

    env = gym.make('CartPole-v1', render_mode="human")
    observation, info = env.reset()
    action = 0
    for _ in range(1000):
        observation, reward, terminated, truncated, info = env.step(action)
        angle = observation[2]
        control = pid_controller.update_step(angle, GOAL, DT)
        if control > 0:
            action = 1
        else:
            action = 0
        print(angle, observation[3])

        if terminated or truncated:
            observation, info = env.reset()

    env.close()
