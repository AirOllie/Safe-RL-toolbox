import datetime
import gym
import os
import os.path as osp
import pickle
import numpy as np
import itertools
import torch
import moviepy.editor as mpy
import cv2
from torch import nn, optim
from algo.Base import Base
from recovery_rl.sac import SAC
from recovery_rl.replay_memory import ReplayMemory, ConstraintReplayMemory
from recovery_rl.MPC import MPC
from recovery_rl.VisualMPC import VisualMPC
from recovery_rl.model import VisualEncoderAttn, TransitionModel, VisualReconModel
from recovery_rl.utils import linear_schedule, recovery_config_setup

from env.make_utils import register_env, make_env

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def torchify(x): return torch.FloatTensor(x).to('cuda')


def npy_to_gif(im_list, filename, fps=4):
    clip = mpy.ImageSequenceClip(im_list, fps=fps)
    clip.write_gif(filename + '.gif')


# Process observation for CNN
def process_obs(obs, env_name):
    if 'extration' in env_name:
        obs = cv2.resize(obs, (64, 48), interpolation=cv2.INTER_AREA)
    im = np.transpose(obs, (2, 0, 1))
    return im

class SQRL(Base):
    def __init__(self, env, env_name='drone_xz', cnn=False, lr=3e-4,
                 updates_per_step=1, start_steps=100, target_update_interval=10, policy='Gaussian', eval=False,
                 gamma=0.999, tau=0.005, alpha=0.2, automatic_entropy_tuning=False, seed=0, batch_size=256,
                 num_steps=1000000, warm_start_num=500, num_eps=1000000, hidden_size=256, replay_size=1000000,
                 task_demos=False, num_task_transitions=10000000, critic_pretraining_steps=3000,
                 pos_fraction=-1, gamma_safe=0.5, eps_safe=0.1, tau_safe=0.0002, safe_replay_size=1000000,
                 num_unsafe_transitions=10000, critic_safe_pretraining_steps=10000,
                 MF_recovery=False, Q_sampling_recovery=False, ctrl_arg=[], override=[], recovery_policy_update_freq=1,
                 vismpc_recovery=False, load_vismpc=False, model_fname='image_maze_dynamics', beta=10, disable_offline_updates=False,
                 disable_online_updates=False, disable_action_relabeling=False, add_both_transitions=False,
                 Q_risk_ablation=False, constraint_reward_penalty=0,
                 DGD_constraints=False, use_constraint_sampling=False, nu=0.01, update_nu=True, nu_schedule=False,
                 nu_start=1e3, nu_end=0, use_LBAC=False, lambda_LBAC=0.01, use_RCPO=False, lambda_RCPO=0.01):

        super(SQRL, self).__init__(env, env_name, cnn, lr,
                 updates_per_step, start_steps, target_update_interval, policy, eval,
                 gamma, tau, alpha, automatic_entropy_tuning, seed, batch_size,
                 num_steps, warm_start_num, num_eps, hidden_size, replay_size,
                 task_demos, num_task_transitions, critic_pretraining_steps,
                 pos_fraction, gamma_safe, eps_safe, tau_safe, safe_replay_size,
                 num_unsafe_transitions, critic_safe_pretraining_steps,
                 MF_recovery, Q_sampling_recovery, ctrl_arg, override, recovery_policy_update_freq,
                 vismpc_recovery, load_vismpc, model_fname, beta, disable_offline_updates,
                 disable_online_updates, disable_action_relabeling, add_both_transitions,
                 Q_risk_ablation, constraint_reward_penalty,
                 DGD_constraints, use_constraint_sampling, nu, update_nu, nu_schedule,
                 nu_start, nu_end)

        self.method_name = 'SQRLv1'
        self.LBAC = use_LBAC
        self.lambda_LBAC =lambda_LBAC
        self.RCPO = use_RCPO
        self.lambda_RCPO = lambda_RCPO
        self.args.method_name = self.method_name
        self.args.LBAC = use_LBAC
        self.args.lambda_LBAC = lambda_LBAC
        self.args.RCPO = use_RCPO
        self.args.lambda_RCPO = lambda_RCPO

        # Logging setup
        self.logdir = os.path.join(
            env_name, '{}_SAC_{}_{}_{}'.format(
                datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
                env_name, policy,
                self.method_name))
        if not os.path.exists(self.logdir):
            os.makedirs(self.logdir)
        print("LOGDIR: ", self.logdir)

        # Experiment setup
        self._experiment_setup(env)


        # Memory
        self.memory = ReplayMemory(replay_size, seed)
        self.recovery_memory = ConstraintReplayMemory(
            safe_replay_size, seed)
        self.all_ep_data = []

        self.total_numsteps = 0
        self.updates = 0
        self.num_constraint_violations = 0

        self.num_viols = 0
        self.num_successes = 0
        self.viol_and_recovery = 0
        self.viol_and_no_recovery = 0
        if not self.eval:  # Get demos
            self.constraint_demo_data, self.task_demo_data, self.obs_seqs, self.ac_seqs, self.constraint_seqs = self._get_offline_data()

        # Get multiplier schedule for RSPO
        if nu_schedule:
            self.nu_schedule = linear_schedule(nu_start,
                                               nu_end,
                                               num_eps)
        else:
            self.nu_schedule = linear_schedule(nu,
                                               nu, 0)

    def _experiment_setup(self, env):
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        recovery_policy = None

        self.env = env
        self.recovery_policy = recovery_policy
        self.env.seed(self.seed)
        self.env.action_space.seed(self.seed)
        agent = self._agent_setup(self.env)
        self.agent = agent


    def _agent_setup(self, env):
        agent = SAC(env.observation_space,
                    env.action_space,
                    self.args,
                    self.logdir,
                    tmp_env=make_env(self.env_name))
        return agent

    def _get_offline_data(self):
        # Get demonstrations
        task_demo_data = None
        obs_seqs = []
        ac_seqs = []
        constraint_seqs = []
        if not self.task_demos:
            if self.env_name == 'reacher':
                constraint_demo_data = pickle.load(
                    open(
                        osp.join("demos", "dvrk_reach",
                                 "constraint_demos.pkl"), "rb"))
                if self.cnn:
                    constraint_demo_data = constraint_demo_data['images']
                else:
                    constraint_demo_data = constraint_demo_data['lowdim']
            elif 'maze' in self.env_name:
                # Maze
                if self.env_name == 'maze':
                    constraint_demo_data = pickle.load(
                        open(
                            osp.join("demos", self.env_name,
                                     "constraint_demos.pkl"), "rb"))
                else:
                    # Image Maze
                    demo_data = pickle.load(
                        open(
                            osp.join("demos", self.env_name,
                                     "demos.pkl"), "rb"))
                    constraint_demo_data = demo_data['constraint_demo_data']
                    obs_seqs = demo_data['obs_seqs']
                    ac_seqs = demo_data['ac_seqs']
                    constraint_seqs = demo_data['constraint_seqs']
            elif 'extraction' in self.env_name:
                # Object Extraction, Object Extraction (Dynamic Obstacle)
                folder_name = self.env_name.split('_env')[0]
                constraint_demo_data = pickle.load(
                    open(
                        osp.join("demos", folder_name, "constraint_demos.pkl"),
                        "rb"))
            else:
                # Navigation 1 and 2
                constraint_demo_data = self.env.transition_function(
                    self.num_unsafe_transitions)
        else:
            if 'extraction' in self.env_name:
                folder_name = self.env_name.split('_env')[0]
                task_demo_data = pickle.load(
                    open(osp.join("demos", folder_name, "task_demos.pkl"),
                         "rb"))
                constraint_demo_data = pickle.load(
                    open(
                        osp.join("demos", folder_name, "constraint_demos.pkl"),
                        "rb"))
                # Get all violations in front to get as many violations as
                # possible
                constraint_demo_data_list_safe = []
                constraint_demo_data_list_viol = []
                for i in range(len(constraint_demo_data)):
                    if constraint_demo_data[i][2] == 1:
                        constraint_demo_data_list_viol.append(
                            constraint_demo_data[i])
                    else:
                        constraint_demo_data_list_safe.append(
                            constraint_demo_data[i])

                constraint_demo_data = constraint_demo_data_list_viol[:int(
                    0.5 * self.num_unsafe_transitions
                )] + constraint_demo_data_list_safe
            else:
                constraint_demo_data, task_demo_data = self.env.transition_function(
                    self.num_unsafe_transitions, task_demos=self.task_demos)
        return constraint_demo_data, task_demo_data, obs_seqs, ac_seqs, constraint_seqs

    def _get_train_rollout(self, i_episode):
        episode_reward = 0
        episode_steps = 0
        done = False
        state = self.env.reset()
        if self.cnn:
            state = process_obs(state, self.env_name)

        train_rollout_info = []
        ep_states = [state]
        ep_actions = []
        ep_constraints = []

        if i_episode % 10 == 0:
            print("SEED: ", self.seed)
            print("LOGDIR: ", self.logdir)

        while not done:
            if len(self.memory) > self.batch_size:
                # Number of updates per step in environment
                for i in range(self.updates_per_step):
                    # Update parameters of all the networks
                    critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = self.agent.update_parameters(
                        self.memory,
                        min(self.batch_size, len(self.memory)),
                        i_episode,
                        safety_critic=self.agent.safety_critic,
                        nu=self.nu_schedule(i_episode))
                    if not self.disable_online_updates and len(
                            self.recovery_memory) > self.batch_size and (
                            self.num_viols + self.num_constraint_violations
                    ) / self.batch_size > self.pos_fraction:
                        self.agent.safety_critic.update_parameters(
                            memory=self.recovery_memory,
                            policy=self.agent.policy,
                            batch_size=self.batch_size,
                            plot=0)
                    self.updates += 1
            # Get action, execute action, and compile step results
            action, real_action, recovery_used = self._get_action(state)
            next_state, reward, done, info = self.env.step(real_action)
            info['recovery'] = recovery_used

            # print(reward)
            if self.cnn:
                next_state = process_obs(next_state, self.env_name)

            if info['constraint']:
                reward -= self.constraint_reward_penalty

            train_rollout_info.append(info)
            episode_steps += 1
            episode_reward += reward
            self.total_numsteps += 1

            mask = float(not done)

            if episode_steps == self.env.max_episode_steps:
                break

            if done:
                print(state, next_state)
            if info['constraint']:
                print(state, next_state)

            # Update buffers
            if not self.disable_action_relabeling:
                self.memory.push(state, action, reward, next_state, mask)
            else:
                # absorbing state
                if info['constraint']:
                    for _ in range(30):
                        self.memory.push(state, real_action, reward, next_state, mask)
                self.memory.push(state, real_action, reward, next_state, mask)

            state = next_state
            ep_states.append(state)
            ep_actions.append(real_action)
            ep_constraints.append([info['constraint']])

        # Get success/violation stats
        if info['constraint']:
            self.num_viols += 1
            if info['recovery']:
                self.viol_and_recovery += 1
            else:
                self.viol_and_no_recovery += 1
        self.num_successes += int(info['success'])


        # Print performance stats
        print("=========================================")
        print("Episode: {}, total numsteps: {}, episode steps: {}, reward: {}".
              format(i_episode, self.total_numsteps, episode_steps,
                     round(episode_reward, 2)))
        print("Num Violations So Far: %d" % self.num_viols)
        print("Violations with Recovery: %d" % self.viol_and_recovery)
        print("Violations with No Recovery: %d" % self.viol_and_no_recovery)
        print("Num Successes So Far: %d" % self.num_successes)
        if info["constraint"]:
            print("Reason: violate")
        elif info["success"]:
            print("Reason: success")
        else:
            print("Reason: timeout")

        return train_rollout_info

    def _get_test_rollout(self, i_episode):
        avg_reward = 0.
        test_rollout_info = []
        state = self.env.reset()

        if 'maze' in self.env_name:
            im_list = [self.env._get_obs(images=True)]
        elif 'extraction' in self.env_name:
            im_list = [self.env.render().squeeze()]

        if self.cnn:
            state = process_obs(state, self.env_name)

        episode_reward = 0
        episode_steps = 0
        done = False
        while not done:
            action, real_action, recovery_used = self._get_action(state,
                                                                 train=False)
            next_state, reward, done, info = self.env.step(real_action)  # Step
            info['recovery'] = recovery_used
            done = done or episode_steps == self.env.max_episode_steps

            if 'maze' in self.env_name:
                im_list.append(self.env._get_obs(images=True))
            elif 'extraction' in self.env_name:
                im_list.append(self.env.render().squeeze())

            if self.cnn:
                next_state = process_obs(next_state, self.env_name)

            test_rollout_info.append(info)
            episode_reward += reward
            episode_steps += 1
            state = next_state

        avg_reward += episode_reward

        if 'maze' in self.env_name or 'extraction' in self.env_name:
            npy_to_gif(im_list, osp.join(self.logdir,
                                         "test_" + str(i_episode)))

        print("----------------------------------------")
        print("Avg. Reward: {}".format(round(avg_reward, 2)))
        print("----------------------------------------")
        return test_rollout_info

    def _dump_logs(self, train_rollouts, test_rollouts):
        data = {"test_stats": test_rollouts, "train_stats": train_rollouts}
        with open(osp.join(self.logdir, "run_stats.pkl"), "wb") as f:
            pickle.dump(data, f)

    def learn(self, num_eps=1000):
        print("================ online training ==================")
        train_rollouts = []
        test_rollouts = []
        for i_episode in range(1,num_eps+1):
            self.agent.i_episode = i_episode
            self.env.i_episode = i_episode
            train_rollout_info = self._get_train_rollout(i_episode)
            train_rollouts.append(train_rollout_info)
            # if i_episode % 10 == 0 and self.eval:
            #     test_rollout_info = self._get_test_rollout(i_episode)
            #     test_rollouts.append(test_rollout_info)
            if i_episode % 500 == 0:
                PATH_policy = "./saved_model/policy_" + str(self.env_name) \
                              + "_" + str(self.method_name) + "_epi" + str(i_episode) + \
                              "_seed" + str(self.seed) + ".pkl"
                PATH_critic = "./saved_model/critic_" + str(self.env_name) \
                              + "_" + str(self.method_name) + "_epi" + str(i_episode) + \
                              "_seed" + str(self.seed) + ".pkl"
                torch.save(self.agent.policy, PATH_policy)
                torch.save(self.agent.critic, PATH_critic)
                PATH_safety_critic = "./saved_model/safety_critic_" + str(self.env_name) \
                                     + "_" + str(self.method_name) + "_epi" + str(i_episode) + \
                                     "_seed" + str(self.seed) + ".pkl"
                torch.save(self.agent.safety_critic.safety_critic, PATH_safety_critic)
        self._dump_logs(train_rollouts, test_rollouts)

    def _get_action(self, state, train=True):
        if self.start_steps > self.total_numsteps and train:
            action = self.env.action_space.sample()  # Sample random action
        elif train:
            action = self.agent.select_action(
                state)  # Sample action from policy
        else:
            action = self.agent.select_action(
                state, eval=True)  # Sample action from policy


        recovery = False
        real_action = np.copy(action)
        return action, real_action, recovery

    def predict(self, obs):
        _, action, _ = self._get_action(obs, train=False)
        return action

    def load_model(self, i_episode):
        PATH_policy = "./saved_model/policy_" + str(self.env_name) \
                      + "_" + str(self.method_name) + "_epi" + str(i_episode) + \
                      "_seed" + str(self.seed) + ".pkl"
        PATH_critic = "./saved_model/critic_" + str(self.env_name) \
                      + "_" + str(self.method_name) + "_epi" + str(i_episode) + \
                      "_seed" + str(self.seed) + ".pkl"
        self.agent.policy = torch.load(PATH_policy)
        self.agent.critic = torch.load(PATH_critic)
        PATH_safety_critic = "./saved_model/safety_critic_" + str(self.env_name) \
                             + "_" + str(self.method_name) + "_epi" + str(i_episode) + \
                             "_seed" + str(self.seed) + ".pkl"
        self.agent.safety_critic.safety_critic = torch.load(PATH_safety_critic)