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


class Sate_rend(gym.Env):
    def __init__(
        self, seed=None, clipped_action=True,
    ):

        # uav质量特性
        self.m = 1
        self.a = 42164e3  # kg
        self.mu = 3.986e14
        self.dt = 0.01

        self._init_state = {"low": [-12, -12, -12, -12], "high": [12, 12, 12, 12]}

        #### reinforcement learning setup  #######
        self._action_clio_warning = False
        self._clipped_action = clipped_action
        # self._init_state = {
        #     "low": [-0.75, 0, -np.pi/4, 0, -1, 0],
        #     "high": [0.75, 1, np.pi/4, 0, 1, 0]
        # }
        self._state_low = np.array([-12, -12, -12, -12])
        self._state_high = np.array([12, 12, 12, 12])
        self._force_low = np.array([-20, -20])
        self._force_high = np.array([20, 20])
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
            self.y = self.state[1]
            self.vx = self.state[2]
            self.vy = self.state[3]
            self.flat_start = self.safe()
        return np.array([self.x, self.y, self.vx, self.vy])

    def unsafe(self):
        done = False
        if np.sqrt(self.x ** 2 + self.y ** 2) >= 9:
            done = True
        if (np.sqrt(self.x ** 2 + self.y ** 2) <= 3) and (self.y >= -np.abs(self.x)):
            done = True
        return done

    def safe(self):
        done = False
        if np.sqrt(self.x ** 2 + self.y ** 2) >= 8:
            done = True
        if (np.sqrt(self.x ** 2 + self.y ** 2) <= 4) and (self.y >= -np.abs(self.x)):
            done = True
        return done

    def barrier(self):
        safe_rate1 = 0
        safe_rate2 = 0
        r_bar = 0
        if np.sqrt(self.x ** 2 + self.y ** 2) >= 8:
            safe_rate1 = (np.sqrt(self.x ** 2 + self.y ** 2)-8) / 1
        if (np.sqrt(self.x ** 2 + self.y ** 2) <= 4) and (self.y >= -np.abs(self.x)):
            safe_rate2 = (4-np.sqrt(self.x ** 2 + self.y ** 2)) / 1
        r_bar = (
            -np.log(1 / (1 + safe_rate1))
            - np.log(1 / (1 + safe_rate2))
        )
        return r_bar

    def success(self):
        done = False
        # if np.abs(self.theta)<1 and  np.sqrt((self.x-self.xt)**2 + (self.z-self.zt)**2) aq nd np.abs(self.dtheta)<1 and np.sqrt(self.vx**2+self.vz**2)<1:
        if (
            np.sqrt(self.x** 2 + self.y** 2) < 0.5
        ):
            done = True
        return done

    def step(self, action):
        ## 控制量坐标系转换
        self.n = np.sqrt(self.mu/self.a**3)
        u = np.clip(action, self.action_space.low, self.action_space.high)
        self.x = (
            self.x
            + self.dt * self.vx
        )
        self.y = (
            self.y
            + self.dt * self.vy
        )
        self.vx = self.vx + self.dt * (2*self.n*self.vy+3*self.n**2*self.x+u[0]/self.m)
        self.vy = self.vy + self.dt * (u[1]/self.m-2*self.n*self.vx)

        self.r = (
            self.x ** 2
            + self.y ** 2
            + self.vx** 2
            + self.vy** 2
            + 10 * self.barrier()
        )
        self.r = -self.r
        self.out = 0

        if self.unsafe():
            self.done = True
            self.r = -1000
        elif self.step_num == 499:
            self.done = True
            self.r = 0
        else:
            self.done = False
        self.step_num = self.step_num + 1
        return self.state, self.r, self.done, self.out


# class Viewer(pyglet.window.Window):
#     def __init__(self, xt, yt, thet, xc, yc, thec, xo1, yo1, theo1):
#         super(Viewer, self).__init__(width=800, height=600, resizable=False, caption='spacecraft docking',
#                                      vsync=False)  # vsync=False to not use the monitor FPS
#         self.set_location(x=100, y=150)
#         pyglet.gl.glClearColor(1, 1, 1, 1)
#         # # 比例系数
#         self.rate = 800/4
#         # # 目标参数 300mm*300mm
#         self.target_info = np.array([self.rate * xt, self.rate * yt, thet])
#         self.target_wide = self.rate * 0.4
#         self.target_high = self.rate * 0.4
#         # 追逐参数 400mm*400mm 位置 3000mm 400mm
#         self.chase_info = np.array([self.rate * xc, self.rate * yc, thec])
#         self.chase_wide = self.rate * 0.1
#         self.chase_high = self.rate * 0.1
#         # 运动障碍物1
#         self.obst1_info = np.array([self.rate * xo1, self.rate * yo1, theo1])
#         self.obst1_wide = self.rate * 0.12
#         self.obst1_high = self.rate * 0.12
#         self.batch = pyglet.graphics.Batch()

#         target_box, chase_box, obst1_box = [0] * 8, [0] * 8, [0] * 8
#         c1, c2, c3 = (249, 86, 86) * 4, (86, 109, 249) * 4, (249, 39, 65) * 4
#         self.target = self.batch.add(4, pyglet.gl.GL_QUADS, None, ('v2f', target_box), ('c3B', c1))
#         self.chase = self.batch.add(4, pyglet.gl.GL_QUADS, None, ('v2f', chase_box), ('c3B', c2))
#         self.obst1 = self.batch.add(4, pyglet.gl.GL_QUADS, None, ('v2f', obst1_box), ('c3B', c3))

#     def render(self, xt, yt, thet, xc, yc, thec, xo1, yo1, theo1):
#         # pyglet.clock.tick()
#         self.target_info = np.array([self.rate * xt, self.rate * yt, thet])
#         self.chase_info = np.array([self.rate * xc, self.rate * yc, thec])
#         self.obst1_info = np.array([self.rate * xo1, self.rate * yo1, theo1])
#         self._update_view()
#         self.switch_to()
#         self.dispatch_events()
#         self.dispatch_event('on_draw')
#         self.flip()

#     def on_draw(self):
#         self.clear()
#         self.batch.draw()

#     def rotation(self,theta,x,y):
#         xr = x*np.cos(theta) - y*np.sin(theta)
#         yr = x*np.sin(theta) + y*np.cos(theta)
#         return xr, yr

#     def _update_view(self):
#         a, b = self.rotation(self.target_info[2],self.target_wide,self.target_high)
#         target_box = (self.target_info[0] - a, self.target_info[1] - b,
#                      self.target_info[0] + b, self.target_info[1] - a,
#                      self.target_info[0] + a, self.target_info[1] + b,
#                      self.target_info[0] - b, self.target_info[1] + a)
#         self.target.vertices = target_box

#         a, b = self.rotation(self.chase_info[2],self.chase_wide,self.chase_high)
#         chase_box = (self.chase_info[0] - a, self.chase_info[1] - b,
#                      self.chase_info[0] + b, self.chase_info[1] - a,
#                      self.chase_info[0] + a, self.chase_info[1] + b,
#                      self.chase_info[0] - b, self.chase_info[1] + a)
#         self.chase.vertices = chase_box

#         a, b = self.rotation(self.obst1_info[2], self.obst1_wide, self.obst1_high)
#         obst1_box = (self.obst1_info[0] - a, self.obst1_info[1] - b,
#                       self.obst1_info[0] + b, self.obst1_info[1] - a,
#                       self.obst1_info[0] + a, self.obst1_info[1] + b,
#                       self.obst1_info[0] - b, self.obst1_info[1] + a)
#         self.obst1.vertices = obst1_box

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
