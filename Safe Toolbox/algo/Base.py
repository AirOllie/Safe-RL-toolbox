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

from recovery_rl.sac import SAC
from recovery_rl.replay_memory import ReplayMemory, ConstraintReplayMemory
from recovery_rl.MPC import MPC
from recovery_rl.VisualMPC import VisualMPC
from recovery_rl.model import VisualEncoderAttn, TransitionModel, VisualReconModel
from recovery_rl.utils import linear_schedule, recovery_config_setup

from env.make_utils import register_env, make_env


class Base(object):
    def __init__(self, env, env_name='drone_xz', cnn=False, lr=3e-4,
                 updates_per_step=1, start_steps=100, target_update_interval=10, policy='Gaussian', eval=True,
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
                 nu_start=1e3, nu_end=0):

        class arguments:
            def __init__(self, env_name='drone_xz', cnn=False, lr=3e-4,
                 updates_per_step=1, start_steps=100, target_update_interval=10, policy='Gaussain', eval=True,
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
                 nu_start=1e3, nu_end=0):

                self.env_name = env_name
                self.cnn = cnn
                self.lr = lr
                self.updates_per_step = updates_per_step
                self.start_steps = start_steps
                self.target_update_interval = target_update_interval
                self.policy = policy
                self.eval = eval
                self.gamma = gamma
                self.tau = tau
                self.alpha = alpha
                self.automatic_entropy_tuning = automatic_entropy_tuning
                self.seed = seed
                self.batch_size = batch_size
                self.num_steps = num_steps
                self.warm_start_num = warm_start_num
                self.num_eps = num_eps
                self.hidden_size = hidden_size
                self.replay_size = replay_size
                self.task_demos = task_demos
                self.num_task_transitions = num_task_transitions
                self.critic_pretraining_steps = critic_pretraining_steps
                self.pos_fraction = pos_fraction
                self.gamma_safe = gamma_safe
                self.eps_safe = eps_safe
                self.tau_safe = tau_safe
                self.safe_replay_size = safe_replay_size
                self.num_unsafe_transitions = num_unsafe_transitions
                self.critic_safe_pretraining_steps = critic_safe_pretraining_steps
                self.MF_recovery = MF_recovery
                self.Q_sampling_recovery = Q_sampling_recovery
                self.ctrl_arg = ctrl_arg
                self.override = override
                self.recovery_policy_update_freq = recovery_policy_update_freq
                self.vismpc_recovery = vismpc_recovery
                self.load_vismpc = load_vismpc
                self.model_fname = model_fname
                self.beta = beta
                self.disable_offline_updates = disable_offline_updates
                self.disable_online_updates = disable_online_updates
                self.disable_action_relabeling = disable_action_relabeling
                self.add_both_transitions = add_both_transitions
                self.Q_risk_ablation = Q_risk_ablation
                self.constraint_reward_penalty = constraint_reward_penalty
                self.DGD_constraints = DGD_constraints
                self.use_constraint_sampling = use_constraint_sampling
                self.nu = nu
                self.update_nu = update_nu
                self.nu_schedule = nu_schedule
                self.nu_start = nu_start
                self.nu_end = nu_end
                self.method_name = None
                self.LBAC = 0
                self.lambda_LBAC = 0
                self.RCPO = 0
                self.lambda_RCPO = 0


        self.args = arguments(env_name, cnn, lr,
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

        self.env_name =env_name
        self.cnn =cnn
        self.lr =lr
        self.updates_per_step =updates_per_step
        self.start_steps =start_steps
        self.target_update_interval =target_update_interval
        self.policy =policy
        self.eval =eval
        self.gamma =gamma
        self.tau =tau
        self.alpha =alpha
        self.automatic_entropy_tuning =automatic_entropy_tuning
        self.seed =seed
        self.batch_size =batch_size
        self.num_steps =num_steps
        self.warm_start_num =warm_start_num
        self.num_eps =num_eps
        self.hidden_size =hidden_size
        self.replay_size =replay_size
        self.task_demos =task_demos
        self.num_task_transitions =num_task_transitions
        self.critic_pretraining_steps =critic_pretraining_steps
        self.pos_fraction =pos_fraction
        self.gamma_safe =gamma_safe
        self.eps_safe =eps_safe
        self.tau_safe =tau_safe
        self.safe_replay_size =safe_replay_size
        self.num_unsafe_transitions =num_unsafe_transitions
        self.critic_safe_pretraining_steps =critic_safe_pretraining_steps
        self.MF_recovery =MF_recovery
        self.Q_sampling_recovery =Q_sampling_recovery
        self.ctrl_arg =ctrl_arg
        self.override =override
        self.recovery_policy_update_freq =recovery_policy_update_freq
        self.vismpc_recovery =vismpc_recovery
        self.load_vismpc =load_vismpc
        self.model_fname =model_fname
        self.beta =beta
        self.disable_offline_updates =disable_offline_updates
        self.disable_online_updates =disable_online_updates
        self.disable_action_relabeling =disable_action_relabeling
        self.add_both_transitions =add_both_transitions
        self.Q_risk_ablation =Q_risk_ablation
        self.constraint_reward_penalty =constraint_reward_penalty
        self.DGD_constraints =DGD_constraints
        self.use_constraint_sampling =use_constraint_sampling
        self.nu =nu
        self.update_nu =update_nu
        self.nu_schedule =nu_schedule
        self.nu_start =nu_start
        self.nu_end =nu_end
