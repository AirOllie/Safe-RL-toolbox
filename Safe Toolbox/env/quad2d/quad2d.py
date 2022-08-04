# #### 画图的教程  #######
import numpy as np
import time
import pyglet
import gym
from gym import logger, spaces
from gym.utils import colorize, seeding

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D, get_test_data
from matplotlib import cm


class Quad2D(gym.Env):
    def __init__(
        self, seed=None, clipped_action=True,
    ):

        # uav质量特性
        self.m = 1  # kg
        self.I = 0.01  # kg
        self.rr = 0.25
        self.g = 9.8
        self.dt = 0.01

        self.xt = 0.5
        self.zt = 0

        #### reinforcement learning setup  #######
        self._action_clio_warning = False
        self._clipped_action = clipped_action
        self._init_state = {
            "low": [-2, 0, 0, -1],
            "high": [-1, 1, 0, 1],
        }
        self._state_low = np.array([-2, -2, -2, -2])
        self._state_high = np.array([2, 2, 2, 2])
        self._force_low = np.array([-0.5, -0.5])
        self._force_high = np.array([0.5, 0.5])
        self.action_space = spaces.Box(
            low=self._force_low, high=self._force_high, dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=-self._state_high, high=self._state_high, dtype=np.float32
        )
        self.seed(seed)

    def seed(self, seed=None):
        """Return random seed."""
        self.np_rando, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        self.step_num = 0
        self.flat_start = True
        while self.flat_start:
            self.state = self.np_rando.uniform(
                low=self._init_state["low"], high=self._init_state["high"]
            )
            self.x = self.state[0]
            self.z = self.state[1]
            self.vx = self.state[2]
            self.vz = self.state[3]
            self.flat_start = self.unsafe()
        return np.array([self.x, self.z, self.vx, self.vz])

    def unsafe(self):
        done = False
        if self.z < -0.3:
            done = True
            # print("a")
        if self.x < -0.5 and self.x > -1.0 and self.z < 0.5 and self.z > -0.4:
            done = True
            # print("b")
        if self.x < 1 and self.x > 0 and self.z < 1.4 and self.z > 0.8:
            done = True
            # print("b")
        if np.sqrt(self.x ** 2 + self.z ** 2 + self.vx ** 2 + self.vz ** 2) >= 5:
            done = True
            # print("c")
        return done

    def barrier(self):
        safe_rate1 = 0
        safe_rate2 = 0
        safe_rate3 = 0
        r_bar = 0
        if self.z < -0.1:
            safe_rate1 = np.abs(self.z + 0.3) / 0.2
        if self.x < -0.4 and self.x > -1.1 and self.z < 0.6 and self.z > -0.5:
            dis_min_1 = np.array(
                [
                    np.abs(self.x + 0.4),
                    np.abs(self.x + 1.1),
                    np.abs(self.z - 0.6),
                    np.abs(self.z + 0.5),
                ]
            )
            safe_rate2 = np.min(dis_min_1) / 0.1
        if self.x < 1.1 and self.x > -0.1 and self.z < 1.5 and self.z > 0.7:
            dis_min_2 = np.array(
                [
                    np.abs(self.x - 1.1),
                    np.abs(self.x + 0.1),
                    np.abs(self.z - 1.5),
                    np.abs(self.z - 0.7),
                ]
            )
            safe_rate3 = np.min(dis_min_2) / 0.1
        # if np.sqrt(self.x**2+self.z**2+self.theta**2+self.vx**2+self.vz**2+self.dtheta**2)>= 4.5:
        #     done = True
        r_bar = (
            -np.log(1 / (1 + safe_rate1))
            - np.log(1 / (1 + safe_rate2))
            - np.log(1 / (1 + safe_rate3))
        )
        return r_bar

    def success(self):
        done = False
        # if np.abs(self.theta)<1 and  np.sqrt((self.x-self.xt)**2 + (self.z-self.zt)**2) aq nd np.abs(self.dtheta)<1 and np.sqrt(self.vx**2+self.vz**2)<1:
        if np.sqrt((self.x - self.xt) ** 2 + (self.z - self.zt) ** 2) < 0.3:
            done = True
        return done

    def step(self, action):
        ## 控制量坐标系转换

        u = np.clip(action, self.action_space.low, self.action_space.high)
        self.x = self.x + self.dt * self.vx
        self.z = self.z + self.dt * self.vz
        self.vx = u[0]
        self.vz = u[1]
        self.state = np.array([self.x, self.z, self.vx, self.vz])

        self.r = (self.x - self.xt) ** 2 + (self.z - self.zt) ** 2
        self.r = -self.r
        self.out = 0

        if self.unsafe():
            self.done = True
            self.r = -1000
        elif self.step_num == 4000:
            self.done = True
            self.r = -1000
        elif self.success():
            self.done = True
            self.r = 0
            self.out = 1
        else:
            self.done = False
        self.step_num = self.step_num + 1
        return self.state, self.r, self.done, self.out


if __name__ == "__main__":
    env = Quad2D()
    num = 0
    action = np.zeros(2)
    for ii in range(1):
        s = env.reset()
        for i in range(6000):
            time.sleep(0.004)
            # env.render()
            action = np.array([4.9, 4.9])
            s, _, done, succ = env.step(action)
            if succ == True:
                num = num + 1
            if done:
                break
    print(num)
