from readline import parse_and_bind
from gym_pybullet_drones.envs.VelocityAviary import VelocityAviary
from gym_pybullet_drones.envs.BaseAviary import DroneModel, Physics
from gym_pybullet_drones.utils.utils import sync, str2bool
import pybullet as p

import time
from gym import Env, utils, spaces
import numpy as np
import numpy.linalg as LA
import gym

import matplotlib
matplotlib.use("TkAgg")


class DroneHover(Env, utils.EzPickle):
    def __init__(self):
        utils.EzPickle.__init__(self)
        simulation_freq_hz = 240
        control_freq_hz = 240
        AGGR_PHY_STEPS = int(simulation_freq_hz / control_freq_hz)
        INIT_XYZS = np.array([[0.5, 0, 0.1]])
        INIT_RPYS = np.array([[0, 0, 0]])
        PHY = Physics.PYB
        self.drone_env = VelocityAviary(
            drone_model=DroneModel.CF2X,
            num_drones=1,
            initial_xyzs=INIT_XYZS,
            initial_rpys=INIT_RPYS,
            physics=PHY,
            neighbourhood_radius=10,
            freq=simulation_freq_hz,
            aggregate_phy_steps=AGGR_PHY_STEPS,
            gui=False,
            record=False,
            obstacles=False,
            user_debug_gui=False,
        )
        
        self.transition_function = get_offline_data
        self._max_episode_steps = 100
        self.action_space = self.get_action_space()
        self.observation_space = self.get_observation_space()
        
        self.obs_drone_state = np.zeros(6,)
        self.obs_drone_state[:3] = self.drone_env.pos
        self.obs_drone_state[3:] = self.drone_env.vel
        
    def step(self, action):
        old_state = self.obs_drone_state.copy()
        drone_action = {str(0): action}
        obs, reward, done, info = self.drone_env.step(drone_action)
        self.obs_drone_state = self.read_obs(obs)
        
        reward = 4*LA.norm(self.obs_drone_state[:3]) + LA.norm(self.obs_drone_state[3:])
        reward = -reward
        
        if self.unsafe():
            done = True
            reward = -1500
        
        info = {
            "constraint": self.unsafe(),
            "reward": reward,
            "state": old_state,
            "next_state": self.obs_drone_state,
            "action": action,
            "success": self.success(),
        }
        return self.obs_drone_state, reward, done, info
        
    def reset(self, random_init=True):
        if random_init:
            self.drone_env.INIT_XYZS = 1 * np.random.random_sample(3).reshape(1, 3)
        else:
            self.drone_env.INIT_XYZS = np.zeros((1, 3))
        obs = self.drone_env.reset()
        return self.read_obs(obs)
    
    def unsafe(self):
        # real val => self.drone_env.pos, self.drone_env.vel, self.drone_env.rpy
        pos = self.obs_drone_state[:3]
        if pos[2] < 0.2:
            return True
        elif LA.norm(pos) > 3:
            return True
        return False
        
    def success(self):
        pos = self.obs_drone_state[:3]
        if LA.norm(pos) < 0.4:
            return True
        else:
            return False
    
    def get_action_space(self):
        # TODO the last term: target_vel=self.SPEED_LIMIT * np.abs(v[3]) * v_unit_vector
        #### Action vector ######### X       Y       Z   fract. of MAX_SPEED_KMH
        act_lower_bound = np.array([-1,     -1,     -1,                        0])
        act_upper_bound = np.array([ 1,      1,      1,                        1])
        return spaces.Box(low=act_lower_bound,high=act_upper_bound,dtype=np.float32)
    
    def get_observation_space(self):        
        #### Observation vector ######### X       Y       Z   VX VY VZ
        obs_lower_bound = np.array([-np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, ])
        obs_upper_bound = np.array([np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, ])
        return spaces.Box(low=obs_lower_bound, high=obs_upper_bound, dtype=np.float32)

    def read_obs(self, obs):
        obs_drone_state = np.zeros(6)
        obs_drone_state[:3] = obs[str(0)]["state"][:3]
        obs_drone_state[3:] = obs[str(0)]["state"][10:13]
        return obs_drone_state
        
        
        
def get_offline_data(num_transitions, task_demos=False, save_rollouts=False):
    print("=============== getting offline data ================")
    env = DroneHover()
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
    register(id="dh-v1", entry_point="drone_hover:DroneHover")
    env = gym.make('dh-v1')
    action = env.action_space.sample()   #  X Y Z   fract. of MAX_SPEED_KMH
    START = time.time()
    obs = env.reset()
    for i in range(100):
        obs, reward, done, info = env.step(action)
        sync(i, START, env.drone_env.TIMESTEP)
        if done:
            obs = env.reset()
        print(info)
