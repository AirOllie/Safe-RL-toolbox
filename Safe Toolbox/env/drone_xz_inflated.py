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



class DroneXZ(Env, utils.EzPickle):
    def __init__(self):
        utils.EzPickle.__init__(self)
        simulation_freq_hz = 240
        control_freq_hz = 240
        AGGR_PHY_STEPS = int(simulation_freq_hz / control_freq_hz)
        INIT_XYZS = np.array([[0, 0, 0.1]])
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
        self._max_episode_steps = 300
        self.action_space = self.get_action_space()
        self.observation_space = self.get_observation_space()
        
        self.obs_drone_state = np.zeros(4,)        
        self.obs_drone_state[0] = self.drone_env.pos.flatten()[0]
        self.obs_drone_state[1] = self.drone_env.pos.flatten()[2]
        self.obs_drone_state[2] = self.drone_env.vel.flatten()[0]
        self.obs_drone_state[3] = self.drone_env.vel.flatten()[2]
        self.i_episode = 0
        
        self.area_left_bottom = {"low": np.array([-2, 0, 0.2]), "high": np.array([-1, 0, 1])}
        self.area_middle_top = {"low": np.array([-2, 0, 1]), "high": np.array([0, 0, 1.8])}
        self.area_right_bottom = {"low": np.array([-0.5, 0, 0.2]), "high": np.array([1, 0, 1.3])}
        
        self.mini_area_left_bottom = {"low": np.array([-1.5, 0, 0.5]), "high": np.array([-1.2, 0, 1])}
        self.mini_area_middle_top = {"low": np.array([-1.5, 0, 1]), "high": np.array([0, 0, 1.3])}
        self.mini_area_right_bottom = {"low": np.array([-0.4, 0, 0.75]), "high": np.array([0.5, 0, 1.25])}

    def step(self, action):
        old_state = self.obs_drone_state.copy()
        action_mag_per = min(LA.norm(action), self.drone_env.SPEED_LIMIT) / self.drone_env.SPEED_LIMIT
        drone_action = {str(0): np.array([action[0], 0, action[1], action_mag_per])}

        for _ in range(25):
            obs, reward, done, info = self.drone_env.step(drone_action)
            self.obs_drone_state = self.read_obs(obs)
            success_flag = self.success()
            unsafe_flag = self.unsafe()
            done = False
            if success_flag or unsafe_flag:
                done = True
                break
    
        x = self.obs_drone_state[0]
        z = self.obs_drone_state[1]

        reward = np.sqrt(4*x**2 + (z-0.5)**2) #6
        reward = -reward
        
        if success_flag:
            reward = 0
        
        if unsafe_flag:
            self.obs_drone_state = old_state  
    
        info = {
            "constraint": unsafe_flag,
            "reward": reward,
            "state": old_state,
            "next_state": self.obs_drone_state,
            "action": action,
            "success": success_flag,
        }
        return self.obs_drone_state, reward, done, info
        
    def reset(self):
        ran_int = np.random.randint(1, 10, 1)
        # change prob. of initial position
        if self.i_episode < 1000:
            if ran_int[0] in [1, 2]:
                init_state = self.mini_area_right_bottom
            elif ran_int[0] in [3, 4, 5]:
                init_state = self.mini_area_left_bottom
            else:
                init_state = self.mini_area_middle_top
        else:
            if ran_int[0] in [1, 2, 3, 4]:
                init_state = self.mini_area_middle_top
            else:
                init_state = self.mini_area_left_bottom

        self.drone_env.INIT_XYZS = np.random.uniform(init_state["low"], init_state["high"]).reshape(1, -1)
        obs = self.drone_env.reset()
        self.obs_drone_state[0] = self.drone_env.pos.flatten()[0]
        self.obs_drone_state[1] = self.drone_env.pos.flatten()[2]
        self.obs_drone_state[2] = self.drone_env.vel.flatten()[0]
        self.obs_drone_state[3] = self.drone_env.vel.flatten()[2]
        return self.obs_drone_state.copy()
    
    def unsafe(self):
        x = self.obs_drone_state[0]
        z = self.obs_drone_state[1]
        if z <= 0.3 or z >= 1.8 or x>1 or x<-2.5: #ddschange
            return True
        if x <= -0.4 and x >= -1.0 and z <= 1.1 and z >= 0.1: #ddschange
            return True
        if x <= 1 and x >= -0.1 and z <= 1.8 and z >= 1.2:  #ddschange
            return True
        # TODO add vel constr.
        if LA.norm(self.obs_drone_state[:2] - np.array([0, 0.5])) > 2.5:
            return True
        if LA.norm(self.obs_drone_state[2:]) > 0.5:
            return True
        return False
        
    def success(self):
        if LA.norm(self.obs_drone_state[:2]-np.array([0, 0.5])) < 0.3:
            return True
        else:
            return False
    
    def get_action_space(self): 
        # vel: x, 
        act_lower_bound = np.array([-0.25, -0.25])
        act_upper_bound = np.array([0.25, 0.25])
        return spaces.Box(low=act_lower_bound,high=act_upper_bound,dtype=np.float32)
    
    def get_observation_space(self):        
        obs_lower_bound = np.array([-3, -3, -1, -1])
        obs_upper_bound = np.array([3, 3, 1, 1])
        return spaces.Box(low=obs_lower_bound, high=obs_upper_bound, dtype=np.float32)

    def read_obs(self, obs):
        obs_drone_state = np.zeros(4)
        obs_drone_state[0] = obs[str(0)]["state"][0]
        obs_drone_state[1] = obs[str(0)]["state"][2]
        obs_drone_state[2] = obs[str(0)]["state"][10]
        obs_drone_state[3] = obs[str(0)]["state"][12]
        return obs_drone_state
        
        
def get_offline_data(num_transitions, task_demos=False, save_rollouts=False):
    print("=============== getting offline data ================")
    env = DroneXZ()
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
            state = next_state.copy()
            if done:
                break
    env.close()
    if save_rollouts:
        return rollouts
    else:
        return transitions
    



if __name__ == "__main__":
    from gym.envs.registration import register
    register(id="dxz-v1", entry_point="drone_xz:DroneXZ")
    env = gym.make('dxz-v1')
    START = time.time()
    obs = env.reset()
    for i in range(100):
        obs = env.reset()
        print(obs)
        print(env.drone_env.SPEED_LIMIT)
        # action = env.action_space.sample()
        # obs, reward, done, info = env.step(action)
        # sync(i, START, env.drone_env.TIMESTEP)
        # print(info)
        # if done:
        #     obs = env.reset()
        # print(info)
