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


class Quad3D(gym.Env):
    def __init__(
        self, seed=None, clipped_action=True,
    ):

        # uav质量特性
        self.m = 1  # kg
        self.g = 9.8
        self.dt = 0.01

        self.xt = 0
        self.zt = 0
        self.yt = 0

        #### reinforcement learning setup  #######
        self._action_clio_warning = False
        self._clipped_action = clipped_action
        self._init_state = {
            "low": [-4, -4, -4, -8, -8, -8, -0.5 * np.pi, -0.5 * np.pi, -0.5 * np.pi],
            "high": [4, 4, 4, 8, 8, 8, 0.5 * np.pi, 0.5 * np.pi, 0.5 * np.pi],
        }
        self._state_low = np.array(
            [-4, -4, -4, -8, -8, -8, -0.5 * np.pi, -0.5 * np.pi, -0.5 * np.pi]
        )
        self._state_high = np.array(
            [4, 4, 4, 8, 8, 8, 0.5 * np.pi, 0.5 * np.pi, 0.5 * np.pi]
        )
        self._force_low = np.array([-100, -50, -50, -50])
        self._force_high = np.array([100, 50, 50, 50])
        self.action_space = spaces.Box(
            low=self._force_low, high=self._force_high, dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=self._state_low, high=self._state_high, dtype=np.float32
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
            self.y = self.state[1]
            self.z = self.state[2]
            self.vx = self.state[3]
            self.vy = self.state[4]
            self.vz = self.state[5]
            self.phi = self.state[6]
            self.theta = self.state[7]
            self.psi = self.state[8]
            self.flat_start = self.safe()
        return np.array(
            [
                self.x,
                self.y,
                self.z,
                self.vx,
                self.vy,
                self.vz,
                self.phi,
                self.theta,
                self.psi,
            ]
        )

    # def render(self, mode="human"):
    #     if self.viewer is None:
    #         self.viewer = Viewer(self.xt, self.yt, self.thet, self.xc, self.yc, self.thec, self.xo1, self.yo1, self.theo1)
    #     self.viewer.render(self.xt, self.yt, self.thet, self.xc, self.yc, self.thec, self.xo1, self.yo1, self.theo1)

    def unsafe(self):
        done = False
        if self.z < -0.3:
            done = True
        if (
            np.sqrt(
                self.x ** 2
                + self.y ** 2
                + self.z ** 2
                + self.vx ** 2
                + self.vy ** 2
                + self.vz ** 2
                + self.phi ** 2
                + self.theta ** 2
                + self.psi ** 2
            )
            >= 3.5
        ):
            done = True
        return done

    def safe(self):
        done = False
        if self.z < 0:
            done = True
        if (
            np.sqrt(
                self.x ** 2
                + self.y ** 2
                + self.z ** 2
                + self.vx ** 2
                + self.vy ** 2
                + self.vz ** 2
                + self.phi ** 2
                + self.theta ** 2
                + self.psi ** 2
            )
            >= 3
        ):
            done = True
        return done

    def barrier(self):
        safe_rate1 = 0
        r_bar = 0
        if self.z < 0:
            safe_rate1 = np.abs(self.z) / 0.3
        # if self.x<-0.4 and self.x>-1.1 and self.z<0.6 and self.z>-0.5:
        #     dis_min_1 = np.array([np.abs(self.x+0.4),np.abs(self.x+1.1),np.abs(self.z-0.6),np.abs(self.z+0.5)])
        #     safe_rate2 = np.min(dis_min_1)/0.1
        # if self.x<1.1 and self.x>-0.1 and self.z<1.5 and self.z>0.7:
        #     dis_min_2 = np.array([np.abs(self.x-1.1),np.abs(self.x+0.1),np.abs(self.z-1.5),np.abs(self.z-0.7)])
        #     safe_rate3 = np.min(dis_min_2)/0.1
        # if np.sqrt(self.x**2+self.z**2+self.theta**2+self.vx**2+self.vz**2+self.dtheta**2)>= 4.5:
        #     done = True
        r_bar = -np.log(1 / (1 + safe_rate1))
        return r_bar

    def success(self):
        done = False
        # if np.abs(self.theta)<1 and  np.sqrt((self.x-self.xt)**2 + (self.z-self.zt)**2) aq nd np.abs(self.dtheta)<1 and np.sqrt(self.vx**2+self.vz**2)<1:
        if (
            np.abs(self.theta) < 1
            and np.sqrt(
                (self.x - self.xt) ** 2
                + (self.y - self.yt) ** 2
                + (self.z - self.zt) ** 2
            )
            < 0.6
        ):
            done = True
        return done

    def step(self, action):
        ## 控制量坐标系转换

        u = np.clip(action, self.action_space.low, self.action_space.high)
        self.x = (
            self.x
            + self.dt * self.vx
            - 0.5 * np.sin(self.theta) * u[0] / self.m * (self.dt ** 2)
        )
        self.y = (
            self.y
            + self.dt * self.vy
            + 0.5
            * np.cos(self.theta)
            * np.sin(self.phi)
            * u[0]
            / self.m
            * (self.dt ** 2)
        )
        self.z = (
            self.z
            + self.dt * self.vz
            + 0.5
            * (-self.g + np.cos(self.theta) * np.cos(self.phi) * u[0] / self.m)
            * (self.dt ** 2)
        )
        self.vx = self.vx - self.dt * np.sin(self.theta) * u[0] / self.m
        self.vy = (
            self.vy + self.dt * np.cos(self.theta) * np.sin(self.phi) * u[0] / self.m
        )
        self.vz = self.vz + self.dt * (
            -self.g + np.cos(self.theta) * np.cos(self.phi) * u[0] / self.m
        )
        self.phi = self.phi + self.dt * u[1]
        self.theta = self.theta + self.dt * u[2]
        self.psi = self.psi + self.dt * u[3]
        self.state = np.array(
            [
                self.x,
                self.y,
                self.z,
                self.vx,
                self.vy,
                self.vz,
                self.phi,
                self.theta,
                self.psi,
            ]
        )

        # self.r = (self.x-self.xt)**2 + (self.y-self.yt)**2 + (self.z-self.zt)**2 + self.theta**2 + self.phi**2 + 10 * self.barrier()
        self.r = (
            (self.x - self.xt) ** 2
            + (self.y - self.yt) ** 2
            + (self.z - self.zt) ** 2
            + self.vx ** 2
            + self.vy ** 2
            + self.vz ** 2
            + self.theta ** 2
            + self.phi ** 2
            + self.psi ** 2
            + 10 * self.barrier()
        )
        self.r = -self.r
        self.out = 0

        if self.unsafe():
            self.done = True
            self.r = -1500
        elif self.step_num == 499:
            self.done = True
            self.out = 1
            self.r = 0
        # elif self.success():
        #     self.done = True
        #     self.r = 0
        #     self.out = 1
        else:
            self.done = False
        self.step_num = self.step_num + 1
        return self.state, self.r, self.done, self.out


if __name__ == "__main__":
    env = Quad3D()
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
