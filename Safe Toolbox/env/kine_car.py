import numpy as np
import numpy.linalg as LA
import time
import gym
from gym import logger, spaces
from gym.utils import colorize, seeding
from gym import Env, utils, spaces

import matplotlib
matplotlib.use("TkAgg")


class KineCar(Env, utils.EzPickle):
    def __init__(self, render=False):
        utils.EzPickle.__init__(self)
        
        self.distur_test = False
        
        # references
        self.v_ref = 0.5  # reference linear velocity 10m/s
        self.omega_ref = 0 # reference angular velocity
        self.dt = 0.05
        
        # state variables
        self.x_e = None
        self.y_e = None
        self.theta_e = None
        self.state = np.zeros(3,)

        #### reinforcement learning setup  #######
        self._force_low = np.array([-0.5, -np.pi/3])
        self._force_high = np.array([0.5, np.pi/3])
        self.action_space = spaces.Box(
            low=self._force_low, high=self._force_high, dtype=np.float32)

        self._state_low = np.array([-2, -2, -np.pi])
        self._state_high = np.array([2, 2, np.pi])
        self.observation_space = spaces.Box(
            low=self._state_low, high=self._state_high, dtype=np.float32)

        self.safe_state_low = np.array([-0.5, -0.5, -np.pi/4])
        self.safe_state_high = np.array([0.5, 0.5, np.pi/4])
        self.safe_observation_space = spaces.Box(
            low=self.safe_state_low, high=self.safe_state_high, dtype=np.float32)
        self.challenge_state_low = np.array([0.4, 0.4, np.pi/4])
        self.challenge_state_high = np.array([0.5, 0.5, np.pi/3])
        self.challenge_observation_space = spaces.Box(
            low=self.challenge_state_low, high=self.challenge_state_high, dtype=np.float32)
        self._max_episode_steps = 100
        self.transition_function = get_offline_data

    def reset(self):
        # self.state = self.challenge_observation_space.sample()
        self.state = self.safe_observation_space.sample()
        self.x_e = self.state[0]
        self.y_e = self.state[1]
        self.theta_e = self.state[2]
        return self.state
        

    def unsafe(self):
        if LA.norm(self.state[:2]) > 1:
            return True
        elif abs(self.state[2]) > np.pi/3:
            return True
        return False

    def success(self):
        return LA.norm(self.state[:2]) < 0.00001
    
    def step(self, action):
        old_state = self.state.copy()
        u = action
        if not self.distur_test:
            u = np.clip(u, self.action_space.low, self.action_space.high)
        self.x_e = self.x_e + self.dt * (
            (u[0]+self.v_ref) * np.cos(self.theta_e)
            - self.v_ref
            + self.omega_ref * self.y_e
        )
        self.y_e = self.y_e + self.dt * (
            (u[0]+self.v_ref) * np.sin(self.theta_e) - self.omega_ref * self.x_e
        )
        self.theta_e = self.theta_e + self.dt * u[1]
        self.state = np.array([self.x_e, self.y_e, self.theta_e])

        if self.theta_e > np.pi:
            self.theta_e = self.theta_e-np.pi
        elif self.theta_e < -np.pi:
            self.theta_e = self.theta_e+np.pi

        reward = 4*np.sqrt(self.x_e ** 2 + self.y_e ** 2) + np.abs(self.theta_e)/np.pi
        reward = -reward

        done = False
        if self.unsafe() or self.success():
            done = True

        info = {
            "constraint": self.unsafe(),
            "reward": reward,
            "state": old_state,
            "next_state": self.state,
            "action": action,
            "success": self.success(),
        }

        return self.state, reward, done, info


def get_offline_data(num_transitions, task_demos=False, save_rollouts=False):
    print("=============== getting offline data ================")
    env = KineCar()
    transitions = []
    rollouts = []
    for i in range(num_transitions // 10):
        rollouts.append([])
        state = env.reset()
        for j in range(10):
            action = env.action_space.sample()
            next_state, reward, done, info = env.step(action)
            constraint = info["constraint"]
            transitions.append(
                (state, action, constraint, next_state, not constraint))
            rollouts[-1].append(
                (state, action, constraint, next_state, not constraint))
            state = next_state
            if done:
                break
    env.close()
    if save_rollouts:
        return rollouts
    else:
        return transitions


if __name__ == "__main__":
    from gym.envs.registration import register
    register(id="car-v0", entry_point="kine_car:KineCar")
    env = gym.make('car-v0')
    action = env.action_space.sample()  # X Y Z   fract. of MAX_SPEED_KMH
    START = time.time()
    obs = env.reset()
    for i in range(100):
        obs, reward, done, info = env.step(action)
        if done:
            obs = env.reset()
        print(info)
