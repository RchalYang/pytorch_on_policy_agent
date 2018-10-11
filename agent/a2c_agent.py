import collections
import copy
import numpy as np
import os.path as osp

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.distributions import Normal

from utils.torch_utils import device, Tensor
import utils.math_utils as math_utils

from .reinforce_agent import ReinforceAgent

class A2CAgent(ReinforceAgent):

    def __init__(self, args, model, optim, env, data_generator, memory, continuous):

        super(A2CAgent,self).__init__(args, model, optim, env, data_generator, memory, continuous)

        self.value_loss_coeff = args.value_loss_coeff
        self.algo="a2c"
        self.step_count = 0

    # def _optimize(self, observations, actions, discounted_rewards):

    #     self.optimizer.zero_grad()
        
    #     observations = Tensor(observations)
    #     actions = Tensor(actions).long()
    #     discounted_rewards = Tensor(discounted_rewards).unsqueeze(1)

    #     dis = self.policy(observations)
    #     dis = Categorical(dis)
    #     log_prob = dis.log_prob(actions).unsqueeze(1)
    #     mean_entropy = dis.entropy().mean()

    #     baseline = self.value(observations).detach()
    #     advantage = discounted_rewards - baseline
    #     # Normalize the advantage
    #     advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

    #     actor_loss = -log_prob * advantage

    #     criterion = nn.MSELoss()
    #     value = self.value(observations)
    #     value_loss = criterion( value, discounted_rewards)

    #     actor_loss = actor_loss.mean() - self.entropy_para * mean_entropy
        
    #     actor_loss.backward()
    #     value_loss.backward()

    #     self.optimizer.step()
    #     self.value_optimizer.step()


    def step(self):

        for batch in self.memory.one_iteration():
            obs, acts, advs, est_rs = batch
            self._optimize( obs, acts, advs, est_rs)

        self.step_time += 1

        # return ave_epi_rew

    def _optimize(self, obs, acts, advs, est_rs):

        self.optim.zero_grad()
        
        obs  = Tensor( obs )
        acts = Tensor( acts )
        advs = Tensor( advs ).unsqueeze(1)
        est_rs = Tensor( est_rs ).unsqueeze(1)

        if self.continuous:
            mean, std, values = self.model( obs )

            dis = Normal(mean, std)
            
            log_prob = dis.log_prob( acts ).sum( -1, keepdim=True )

            ent = dis.entropy().sum( -1, keepdim=True )

        else:

            probs, values = self.model(obs)

            dis = F.softmax( probs, dim = 1 )

            acts = acts.long()

            dis = Categorical( probs )

            log_prob = dis.log_prob( acts ).unsqueeze(1)

            ent = dis.entropy()


        # Normalize the advantage
        advs = (advs - advs.mean()) / (advs.std() + 1e-8)

        policy_loss = -log_prob * advs
        policy_loss = policy_loss.mean() - self.entropy_para * ent.mean()

        criterion = nn.MSELoss()
        critic_loss = criterion( values, est_rs )

        self.writer.add_scalar( "Training/Critic_Loss", critic_loss.item(), self.step_count )
        loss = policy_loss + self.value_loss_coeff * critic_loss

        loss.backward()

        self.optim.step()

        self.step_count += 1
