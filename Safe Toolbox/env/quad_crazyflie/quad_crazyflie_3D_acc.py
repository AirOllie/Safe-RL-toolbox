# #### 画图的教程  #######
from types import TracebackType
import numpy as np
import time
import pyglet
import gym
from gym import logger, spaces
from gym.utils import colorize, seeding


import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import math
from scipy.integrate import odeint
from mpl_toolkits.mplot3d import Axes3D

import os
import time
import argparse
from datetime import datetime
import pdb
import math
import random
import numpy as np
import pybullet as p
import matplotlib.pyplot as plt

from gym_pybullet_drones.envs.BaseAviary import DroneModel, Physics
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.envs.VisionAviary import VisionAviary
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.control.SimplePIDControl import SimplePIDControl
from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.utils.utils import sync, str2bool
from gym_pybullet_drones.envs.VelocityAviary import VelocityAviary


class Crazyflie(gym.Env):
    def __init__(self):

        parser = argparse.ArgumentParser(
            description="Velocity control example using VelocityAviary"
        )
        parser.add_argument(
            "--drone",
            default="cf2x",
            type=DroneModel,
            help="Drone model (default: CF2X)",
            metavar="",
            choices=DroneModel,
        )
        parser.add_argument(
            "--gui",
            default=True,
            type=str2bool,
            help="Whether to use PyBullet GUI (default: True)",
            metavar="",
        )
        parser.add_argument(
            "--record_video",
            default=False,
            type=str2bool,
            help="Whether to record a video (default: False)",
            metavar="",
        )
        parser.add_argument(
            "--plot",
            default=False,
            type=str2bool,
            help="Whether to plot the simulation results (default: True)",
            metavar="",
        )
        parser.add_argument(
            "--user_debug_gui",
            default=False,
            type=str2bool,
            help="Whether to add debug lines and parameters to the GUI (default: False)",
            metavar="",
        )
        parser.add_argument(
            "--aggregate",
            default=True,
            type=str2bool,
            help="Whether to aggregate physics steps (default: False)",
            metavar="",
        )
        parser.add_argument(
            "--obstacles",
            default=False,
            type=str2bool,
            help="Whether to add obstacles to the environment (default: True)",
            metavar="",
        )
        parser.add_argument(
            "--simulation_freq_hz",
            default=240,
            type=int,
            help="Simulation frequency in Hz (default: 240)",
            metavar="",
        )
        parser.add_argument(
            "--control_freq_hz",
            default=240,
            type=int,
            help="Control frequency in Hz (default: 48)",
            metavar="",
        )
        parser.add_argument(
            "--duration_sec",
            default=50,
            type=int,
            help="Duration of the simulation in seconds (default: 5)",
            metavar="",
        )
        self.ARGS = parser.parse_args()

        #### Initialize the simulation #############################
        self.INIT_XYZS = np.array([[0, 0, 0.1],])
        self.INIT_RPYS = np.array([[0, 0, 0],])
        self.AGGR_PHY_STEPS = (
            int(self.ARGS.simulation_freq_hz / self.ARGS.control_freq_hz)
            if self.ARGS.aggregate
            else 1
        )
        self.PHY = Physics.PYB

        #### Create the environment ################################

        action = {str(0): np.array([0, 0, 0, 0])}
        START = time.time()
        high_s = np.array([4, 4, 4, 0.5, 0.5, 0.5,])
        high_a = np.array([5, 5, 5])
        self.action_space = spaces.Box(low=-high_a, high=high_a, dtype=np.float32)
        high_v = np.array([1, 1, 1])
        self.v_space = spaces.Box(low=-high_v, high=high_v, dtype=np.float32)
        self.observation_space = spaces.Box(-high_s, high_s, dtype=np.float32)
        self.a_bound = self.action_space.high
        self.modify_action_scale = True
        self.seed()
        self.viewer = None
        self.state = None
        self.steps_beyond_done = None

    def seed(self, seed=None):
        """Return random seed."""
        self.np_rando, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        self.step_num = 0
        self.flat_start = True
        self._init_state = {
            "low": [-3, -3, 0.5],
            "high": [3, 3, 3.5],
        }
        self.flat_start = True
        while self.flat_start:
            self.INIT_XYZS = self.np_rando.uniform(
                low=self._init_state["low"], high=self._init_state["high"]
            )
            self.flat_start = self.inital()
        self.INIT_XYZS = np.array([self.INIT_XYZS,])
        self.env = VelocityAviary(
            drone_model=self.ARGS.drone,
            num_drones=1,
            initial_xyzs=self.INIT_XYZS,
            initial_rpys=self.INIT_RPYS,
            physics=self.PHY,
            neighbourhood_radius=10,
            freq=self.ARGS.simulation_freq_hz,
            aggregate_phy_steps=self.AGGR_PHY_STEPS,
            gui=self.ARGS.gui,
            record=self.ARGS.record_video,
            obstacles=self.ARGS.obstacles,
            user_debug_gui=self.ARGS.user_debug_gui,
        )

        #### Obtain the PyBullet Client ID from the environment ####
        self.PYB_CLIENT = self.env.getPyBulletClient()
        self.DRONE_IDS = self.env.getDroneIds()

        #### Run the simulation ####################################
        self.CTRL_EVERY_N_STEPS = int(
            np.floor(self.env.SIM_FREQ / self.ARGS.control_freq_hz)
        )
        u = {str(0): np.array([0, 0, 0, 1,])}
        obs, reward, done, info = self.env.step(u)
        state = obs[str(0)]["state"]
        self.x = state[0]
        self.y = state[1]
        self.z = state[2]
        self.vx = state[10]
        self.vy = state[11]
        self.vz = state[12]
        self.roll = state[7]
        self.pitch = state[8]
        self.yaw = state[9]

        self.state = np.array(
            [self.x, self.y, (self.z - 0.5), self.vx, self.vy, self.vz,]
        )
        return self.state

    def unsafe(self):
        done = False
        if self.z < 0.2:
            done = True
        if (
            np.sqrt(
                self.x ** 2
                + self.y ** 2
                + (self.z - 0.5) ** 2
                + self.vx ** 2
                + self.vy ** 2
                + self.vz ** 2
            )
            >= 4.5
        ):
            done = True
        return done

    def inital(self):
        done = False
        if (
            np.sqrt(
                self.INIT_XYZS[0] ** 2
                + self.INIT_XYZS[1] ** 2
                + (self.INIT_XYZS[2] - 0.5) ** 2
            )
            >= 4
        ):
            done = True
        return done

    def step(self, action):
        ## 控制量坐标系转换
        action = np.clip(action, self.action_space.low, self.action_space.high)
        action[0] = self.vx + 0.1 * action[0]
        action[1] = self.vy + 0.1 * action[1]
        action[2] = self.vz + 0.1 * action[2]
        action = np.clip(action, self.v_space.low, self.v_space.high)
        u = {
            str(0): np.array(
                [
                    action[0],
                    action[1],
                    action[2],
                    np.sqrt(action[0] ** 2 + action[1] ** 2 + action[2] ** 2),
                ]
            )
        }
        for i in range(5):
            obs, reward, done, info = self.env.step(u)
        state = obs[str(0)]["state"]
        self.x = state[0]
        self.y = state[1]
        self.z = state[2]
        self.vx = state[10]
        self.vy = state[11]
        self.vz = state[12]
        self.roll = state[7]
        self.pitch = state[8]
        self.yaw = state[9]

        self.state = np.array(
            [self.x, self.y, (self.z - 0.5), self.vx, self.vy, self.vz,]
        )

        self.r = 4 * np.sqrt(self.x ** 2 + self.y ** 2 + (self.z - 0.5) ** 2) + np.sqrt(
            self.vx ** 2 + self.vy ** 2 + self.vz ** 2
        )
        self.r = -self.r
        self.out = 0

        if self.unsafe():
            self.done = True
            self.r = -1500
            # print("aaa")
        elif self.step_num == 2999:
            self.done = True
            self.out = 1
            self.r = 0
            # print("bbb")
        else:
            self.done = False
        self.step_num = self.step_num + 1

        if self.done:
            self.env.close()
        return self.state, self.r, self.done, self.out


if __name__ == "__main__":
    env = Crazyflie()
    num = 0
    action = np.zeros(3)
    for ii in range(3):
        s = env.reset()
        for i in range(6000):
            # time.sleep(0.01)
            action = np.array([0, 1, 0])
            s, _, done, succ = env.step(action)
            # print(s)
            if succ == True:
                num = num + 1
            if done:
                break
