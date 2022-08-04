from email import policy
import torch
import gym
from env.make_utils import register_env, make_env
import numpy.linalg as LA
import matplotlib.pyplot as plt
import numpy as np
import eva_agents
import time
from plotting import plotting_utils


class Evaluator():
    def __init__(self, env_name, distur_test=False):
        register_env(env_name)
        self.env = make_env(env_name)
        self.num_success = 0
        self.num_violation = 0
        self.agent = None
        self.get_action_time = 0
        self.distur_test = distur_test

    def evaluation(self, num_episodes):
        self.num_success = 0
        self.num_violation = 0
        all_traj_info = []
        for i_episode in range(num_episodes):
            all_traj_info.append(self.get_test_rollout(i_episode))
        print("Success Rate: {}".format(self.num_success/num_episodes))
        print("Violation Rate: {}".format(self.num_violation/num_episodes))
        return all_traj_info

    def get_test_rollout(self, i_episode):
        avg_reward = 0.
        test_rollout_info = []
        state = self.env.reset()

        episode_reward = 0
        episode_steps = 0
        done = False
        
        if self.distur_test:
            self.env.distur_test = True
        cnt = 0
        while not done:
            cnt += 1
            now = time.time()
            action = self.agent.get_action(state)
            if self.distur_test and cnt%20 == 0:
                # action += np.random.normal(-1, 1, size=action.shape)
                action_sign = np.sign(action)
                action[0] = -11 * action_sign[0]
                action[1] = -2*np.pi * action_sign[1]
            
            self.get_action_time = max(time.time() - now, self.get_action_time)
            next_state, reward, done, info = self.env.step(action)  # Step
            done = done or episode_steps == self.env._max_episode_steps
            test_rollout_info.append(info)
            episode_reward += reward
            episode_steps += 1
            state = next_state
        avg_reward += episode_reward
        # print("----------------------------------------")
        # print("Test Rollout: {}".format(i_episode))
        # print("Avg. Reward: {}".format(round(avg_reward, 2)))
        # print("Final state: {}".format(state))
        # print("Get action time: {}ms".format(self.get_action_time * 1000))
        if info["constraint"]:
            # print("Terminate: Violation")
            self.num_violation += 1
        elif info["success"]:
            # print("Terminate: Success")
            self.num_success += 1
        # else:
        #     print("Terminate: Timeout")
        # print("----------------------------------------")
        return test_rollout_info


def plot_state_multi_traj(all_traj_info, axs, alg):
    pos_err_multi = []
    theta_err_multi = []
    for test_rollout_info in all_traj_info:
        num_steps = len(test_rollout_info)
        x_e = []
        y_e = []
        theta_e = []
        for i in range(num_steps):
            x_e.append(test_rollout_info[i]["state"][0])
            y_e.append(test_rollout_info[i]["state"][1])
            theta_e.append(test_rollout_info[i]["state"][2])
        x_e = np.array(x_e)
        y_e = np.array(y_e)
        pos_e = LA.norm(np.array([x_e, y_e]), axis=0)
        theta_e = np.array(theta_e)
        pos_err_multi.append(pos_e)
        theta_err_multi.append(theta_e)
    data = [pos_err_multi, theta_err_multi]
    for i in range(len(data)):
        err_mean, err_lb, err_ub = plotting_utils.get_stats(np.array(data[i]))
        axs[i].fill_between(
            range(err_mean.shape[0]),
            err_lb,
            err_ub,
            alpha = 0.25,
            label = alg
        )
        axs[i].plot(err_mean)
   

policy_map = {"LBAC": ["policy_kine_car_LBAC_5"],
              "RSPO": ["policy_kine_car_RSPO_epi2000_seed32"],
              "RCPO": ["policy_kine_car_RCPO_epi2000_seed35"],
              "SQRL": ["policy_kine_car_SQRL_epi1500_seed41"],
              "MPC": []}

policy_map = {"RSPO": ["policy_kine_car_RSPO_epi2000_seed32"]}

if __name__ == "__main__":
    fig, axs = plt.subplots(2, figsize=(16, 8))
    evaluator = Evaluator(env_name="kine_car", distur_test=True)
    dir = "/home/pw/Shaohang/safe_rl/py36_code/Actor_critic_with_safety/car_model_final/"
    dir = "/home/pw/Shaohang/Dropbox/reading/saved_model/car/New_policy/"
    for alg in policy_map:
        print("Evaluating: {}".format(alg))
        if alg == "MPC":
            policy_map[alg].append(eva_agents.Car_NMPC_Agent(control_freq=10))
        else:
            PATH = dir + policy_map[alg][0] + ".pkl"
            agent = eva_agents.RL_Agent(PATH)
            policy_map[alg].append(agent)
        evaluator.agent = policy_map[alg][-1]
        all_traj_info = evaluator.evaluation(num_episodes=100)
    #     plot_state_multi_traj(all_traj_info, axs, alg)
    # for i in range(axs.shape[0]):
    #     ax = axs[i]
    #     if i == 0:
    #         ax.axhline(y=1, color='red', linestyle='--')
    #         ax.set_ylabel("Position Error", fontsize=16)
    #         ax.set_ylim(top=1.5)
    #     elif i == 1:
    #         ax.axhline(y=np.pi/3, color='red', linestyle='--')
    #         ax.axhline(y=-np.pi/3, color='red', linestyle='--')
    #         ax.set_ylabel("Theta Error", fontsize=16)
    #     ax.axhline(y=0, color='black', linestyle='--')
    #     ax.legend(loc="upper right", fontsize=12, frameon=False)
    #     ax.set_xlabel("Time", fontsize=16)
    # plt.show()
