'''
Built on on SAC implementation from 
https://github.com/pranz24/pytorch-soft-actor-critic
'''

import os
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image
import os.path as osp
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
from dotmap import DotMap
import cv2

from recovery_rl.utils import soft_update, hard_update
from recovery_rl.model import GaussianPolicy, QNetwork, DeterministicPolicy, QNetworkCNN, LNetwork, \
    GaussianPolicyCNN, QNetworkConstraint, QNetworkConstraintCNN, DeterministicPolicyCNN
from recovery_rl.qrisk import QRiskWrapper

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


class SAC(object):
    def __init__(self,
                 observation_space,
                 action_space,
                 args,
                 logdir,
                 im_shape=None,
                 tmp_env=None):

        self.torchify = lambda x: torch.FloatTensor(x).to(self.device)
        # Parameters for SAC
        self.gamma = args.gamma
        self.tau = args.tau
        self.alpha = args.alpha
        self.env_name = args.env_name
        self.logdir = logdir
        self.policy_type = args.policy
        self.target_update_interval = args.target_update_interval
        self.automatic_entropy_tuning = args.automatic_entropy_tuning
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.updates = 0
        self.cnn = args.cnn
        self.warm_start_num = args.warm_start_num
        if im_shape:
            observation_space = im_shape
        # RRL specific parameters
        self.gamma_safe = args.gamma_safe
        self.eps_safe = args.eps_safe
        # Parameters for comparisons
        self.DGD_constraints = args.DGD_constraints
        self.nu = args.nu
        self.update_nu = args.update_nu
        self.use_constraint_sampling = args.use_constraint_sampling
        self.log_nu = torch.tensor(np.log(self.nu),
                                   requires_grad=True,
                                   device=self.device)
        self.nu_optim = Adam([self.log_nu], lr=0.1 * args.lr)
        self.RCPO = args.RCPO
        self.LBAC = args.LBAC
        self.i_episode = 0
        self.lambda_RCPO = args.lambda_RCPO
        self.lambda_LBAC = args.lambda_LBAC
        self.log_lambda_RCPO = torch.tensor(np.log(self.lambda_RCPO),
                                            requires_grad=True,
                                            device=self.device)
        self.lambda_RCPO_optim = Adam(
            [self.log_lambda_RCPO],
            lr=0.1 * args.lr)  # Make lambda update slower for stability

        # LBAC
        self.log_lambda_LBAC = torch.tensor(np.log(self.lambda_LBAC),
                                            requires_grad=True,
                                            device=self.device)
        self.lambda_LBAC_optim = Adam(
            [self.log_lambda_LBAC],
            lr=0.1 * args.lr)  # Make lambda update slower for stability

        # SAC setup
        if args.LBAC:
            self.critic = QNetwork(observation_space.shape[0], action_space.shape[0],
                                   args.hidden_size).to(device=self.device)
            self.critic_target = QNetwork(
                observation_space.shape[0], action_space.shape[0], args.hidden_size).to(device=self.device)
        else:
            if args.cnn:
                self.critic = QNetworkCNN(observation_space, action_space.shape[0],
                                          args.hidden_size,
                                          args.env_name).to(device=self.device)
                self.critic_target = QNetworkCNN(
                    observation_space, action_space.shape[0], args.hidden_size,
                    args.env_name).to(device=self.device)
            else:
                self.critic = QNetwork(observation_space.shape[0],
                                       action_space.shape[0],
                                       args.hidden_size).to(device=self.device)
                self.critic_target = QNetwork(
                    observation_space.shape[0], action_space.shape[0],
                    args.hidden_size).to(device=self.device)
        self.critic_optim = Adam(self.critic.parameters(), lr=args.lr)
        hard_update(self.critic_target, self.critic)
        if self.policy_type == "Gaussian":
            # Target Entropy = −dim(A)
            if self.automatic_entropy_tuning is True:
                self.target_entropy = -torch.prod(
                    torch.Tensor(action_space.shape).to(self.device)).item()
                self.log_alpha = torch.zeros(1,
                                             requires_grad=True,
                                             device=self.device)
                self.alpha_optim = Adam([self.log_alpha], lr=args.lr)

            if args.cnn:
                self.policy = GaussianPolicyCNN(observation_space,
                                                action_space.shape[0],
                                                args.hidden_size,
                                                args.env_name,
                                                action_space).to(self.device)
            else:
                self.policy = GaussianPolicy(observation_space.shape[0],
                                             action_space.shape[0],
                                             args.hidden_size,
                                             action_space).to(self.device)

            self.policy_optim = Adam(self.policy.parameters(), lr=args.lr)

        else:
            self.alpha = 0
            self.automatic_entropy_tuning = False
            assert not args.cnn
            self.policy = DeterministicPolicy(observation_space.shape[0],
                                              action_space.shape[0],
                                              args.hidden_size,
                                              action_space).to(self.device)
            self.policy_optim = Adam(self.policy.parameters(), lr=args.lr)
        # Initialize safety critic
        self.safety_critic = QRiskWrapper(observation_space,
                                          action_space,
                                          args.hidden_size,
                                          logdir,
                                          args,
                                          tmp_env=tmp_env)
        
        
        ## retrain policy and critic #ddschange
        # self.policy = torch.load("saved_model/policy_drone_xz_SQRLv1_epi2500_seed44.pkl")
        # self.critic = torch.load("saved_model/critic_drone_xz_SQRLv1_epi2500_seed44.pkl")
        # self.policy = torch.load("saved_model/policy_drone_xz_RSPO_epi2500_seed85.pkl")
        # self.critic = torch.load("saved_model/critic_drone_xz_RSPO_epi2500_seed85.pkl")
        # self.policy = torch.load("saved_model/policy_drone_xz_LBACv1_epi2500_seed53.pkl")
        # self.critic = torch.load("saved_model/critic_drone_xz_LBACv1_epi2500_seed53.pkl")
        # self.policy = torch.load("saved_model/policy_drone_xz_LBACv1_epi3000_seed42.pkl")
        # self.critic = torch.load("saved_model/critic_drone_xz_LBACv1_epi3000_seed42.pkl")
        # self.policy = torch.load("saved_model/policy_drone_xz_LBAC_1.pkl")
        # self.critic = torch.load("saved_model/critic_drone_xz_LBAC_1.pkl")
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        

    def select_action(self, state, eval=False):
        '''
            Get action from current task policy
        '''
        # torch.manual_seed(0)
        # torch.cuda.manual_seed_all(0)
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        # Action Sampling for SQRL
        if self.use_constraint_sampling:
            self.safe_samples = 100  # TODO: don't hardcode
            if not self.cnn:
                state_batch = state.repeat(self.safe_samples, 1)
            else:
                state_batch = state.repeat(self.safe_samples, 1, 1, 1)
            pi, log_pi, _ = self.policy.sample(state_batch)
            max_qf_constraint_pi = self.safety_critic.get_value(
                state_batch, pi)

            thresh_idxs = (max_qf_constraint_pi <= self.eps_safe).nonzero()[:, 0]
            # Note: these are auto-normalized
            thresh_probs = torch.exp(log_pi[thresh_idxs])
            thresh_probs = thresh_probs.flatten()

            if list(thresh_probs.size())[0] == 0:
                min_q_value_idx = torch.argmin(max_qf_constraint_pi)
                action = pi[min_q_value_idx, :].unsqueeze(0)
            else:
                prob_dist = torch.distributions.Categorical(thresh_probs)
                sampled_idx = prob_dist.sample()
                action = pi[sampled_idx, :].unsqueeze(0)
        # Action Sampling for all other algorithms
        else:
            if eval is False:
                action, _, _ = self.policy.sample(state)
            else:
                _, _, action = self.policy.sample(state)
        return action.detach().cpu().numpy()[0]

    def update_parameters(self,
                          memory,
                          batch_size,
                          updates,
                          nu=None,
                          safety_critic=None):
        '''
        Train task policy and associated Q function with experience in replay buffer (memory)
        '''
        if nu is None:
            nu = self.nu
        # Sample a batch from memory
        state_batch, action_batch, reward_batch, next_state_batch, mask_batch = memory.sample(
            batch_size=batch_size)

        state_batch = torch.FloatTensor(state_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(
            self.device).unsqueeze(1)
        mask_batch = torch.FloatTensor(mask_batch).to(self.device).unsqueeze(1)

        with torch.no_grad():
            next_state_action, next_state_log_pi, _ = self.policy.sample(
                next_state_batch)
            qf1_next_target, qf2_next_target = self.critic_target(
                next_state_batch, next_state_action)
            min_qf_next_target = torch.min(
                qf1_next_target,
                qf2_next_target) - self.alpha * next_state_log_pi
            next_q_value = reward_batch + mask_batch * self.gamma * (
                min_qf_next_target)
            if self.RCPO:
                qsafe_batch = torch.max(
                    *safety_critic(state_batch, action_batch))
                next_q_value -= self.lambda_RCPO * qsafe_batch
        qf1, qf2 = self.critic(
            state_batch, action_batch
        )  # Two Q-functions to mitigate positive bias in the policy improvement step
        qf1_next, qf2_next = self.critic_target(
            next_state_batch, next_state_action)
        if self.LBAC and self.i_episode > self.warm_start_num:
        # if self.LBAC:
            # qf1_loss = F.mse_loss(
            #     qf1, next_q_value
            # )  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
            l_delta_1 = torch.mean(
                - qf1_next.min() + qf1.detach() + 0.000001)
            # qf1_loss = 0.5 * ((qf1 - next_q_value) ** 2 ).mean()
            qf1_loss = 0.5 * ((qf1 - next_q_value) ** 2
                              + self.lambda_LBAC * l_delta_1).mean()
            # qf1_loss += F.mse_loss(
            #     qf1, next_q_value
            # )  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
            l_delta_2 = torch.mean(
                - qf2_next.min() + qf2.detach() + 0.000001)
            # qf2_loss = 0.5 * ((qf2 - next_q_value) ** 2).mean()
            qf2_loss = 0.5 * ((qf2 - next_q_value) ** 2
                              + self.lambda_LBAC * l_delta_2).mean()
            l_delta = l_delta_1 + l_delta_2
            # print(self.lambda_LBAC)
        else:
            qf1_loss = F.mse_loss(
                qf1, next_q_value
            )  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
            # qf1_loss = 0.5 * ((qf1 - next_q_value) ** 2).mean()
            qf2_loss = F.mse_loss(
                qf2, next_q_value
            )  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
            # qf2_loss = 0.5 * ((qf2 - next_q_value) ** 2).mean()
        pi, log_pi, _ = self.policy.sample(state_batch)

        qf1_pi, qf2_pi = self.critic(state_batch, pi)
        # qf1_pi = torch.ones(256).to(device)
        # qf2_pi = torch.ones(256).to(device)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)

        sqf1_pi, sqf2_pi = self.safety_critic(state_batch, pi)
        max_sqf_pi = torch.max(sqf1_pi, sqf2_pi)

        if self.DGD_constraints:
            policy_loss = (
                (self.alpha * log_pi) + nu * (max_sqf_pi - self.eps_safe) -
                1. * min_qf_pi
            ).mean()  # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]
        else:
            policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean(
            )  # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]

        self.critic_optim.zero_grad()
        (qf1_loss + qf2_loss).backward()
        self.critic_optim.step()

        self.policy_optim.zero_grad()
        # torch.autograd.set_detect_anomaly(True)
        policy_loss.backward()
        self.policy_optim.step()

        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha *
                           (log_pi + self.target_entropy).detach()).mean()

            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()

            self.alpha = self.log_alpha.exp()
            alpha_tlogs = self.alpha.clone()  # For TensorboardX logs
        else:
            alpha_loss = torch.tensor(0.).to(self.device)
            alpha_tlogs = torch.tensor(self.alpha)  # For TensorboardX logs

        # Optimize nu for LR
        if self.update_nu:
            nu_loss = (self.log_nu *
                       (self.eps_safe - max_sqf_pi).detach()).mean()
            self.nu_optim.zero_grad()
            nu_loss.backward()
            self.nu_optim.step()
            self.nu = self.log_nu.exp()

        # Optimize lambda for RCPO
        if self.RCPO:
            lambda_RCPO_loss = (self.log_lambda_RCPO *
                                (self.eps_safe - qsafe_batch).detach()).mean()
            self.lambda_RCPO_optim.zero_grad()
            lambda_RCPO_loss.backward()
            self.lambda_RCPO_optim.step()
            self.lambda_RCPO = self.log_lambda_RCPO.exp()

        # Optimize lambda for LBAC
        if self.LBAC and self.i_episode > self.warm_start_num:
        # if self.LBAC:
            lambda_LBAC_loss = (self.log_lambda_LBAC *
                                l_delta.detach()).mean()
            self.lambda_LBAC_optim.zero_grad()
            lambda_LBAC_loss.backward()
            self.lambda_LBAC_optim.step()
            self.lambda_LBAC = self.log_lambda_LBAC.exp()

        if updates % self.target_update_interval == 0:
            soft_update(self.critic_target, self.critic, self.tau)

        return qf1_loss.item(), qf2_loss.item(), policy_loss.item(
        ), alpha_loss.item(), alpha_tlogs.item()
