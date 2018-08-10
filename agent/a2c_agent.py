import collections
import copy
import numpy as np
import os.path as osp

import torch
import torch.nn as nn
from torch.distributions import Categorical

from utils.torch_utils import device, Tensor
import utils.math_utils as math_utils

from .base_agent import BaseAgent

class A2CAgent(BaseAgent):
    def __init__(self,
                env,
                policy,
                policy_optimizer,
                value,
                value_optimizer,
                episodes,
                gamma,
                entropy_para,
                batch_size,
                tau
                ):


        super(A2CAgent,self).__init__(env,
                policy,    policy_optimizer,
                episodes,  gamma,
                entropy_para, batch_size,
                tau)

        self.value        = value.to(device)
        self.value_optimizer = value_optimizer
        self.algo="a2c"

    def _optimize(self, observations, actions, discounted_rewards):

        self.optimizer.zero_grad()
        self.value_optimizer.zero_grad()
        
        observations = Tensor(observations)
        actions = Tensor(actions).long()
        discounted_rewards = Tensor(discounted_rewards).unsqueeze(1)

        dis = self.policy(observations)
        dis = Categorical(dis)
        log_prob = dis.log_prob(actions).unsqueeze(1)
        mean_entropy = dis.entropy().mean()

        baseline = self.value(observations).detach()
        advantage = discounted_rewards - baseline
        # Normalize the advantage
        advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

        actor_loss = -log_prob * advantage

        criterion = nn.MSELoss()
        value = self.value(observations)
        value_loss = criterion( value, discounted_rewards)

        actor_loss = actor_loss.mean() - self.entropy_para * mean_entropy
        
        actor_loss.backward()
        value_loss.backward()

        self.optimizer.step()
        self.value_optimizer.step()

    def load_model(self, prefix):
        policy_file_name="{}_policy_latest_model.pth".format(self.algo)
        value_file_name="{}_value_latest_model.pth".format(self.algo)
        policy_path=osp.join(prefix,policy_file_name)
        value_path =osp.join(prefix,value_file_name)
        self.policy.load_state_dict(torch.load(policy_path))
        self.value.load_state_dict(torch.load(value_path))

    def snapshot(self, prefix):
        policy_file_name="{}_policy_latest_model.pth".format(self.algo)
        value_file_name="{}_value_latest_model.pth".format(self.algo)
        policy_path=osp.join(prefix,policy_file_name)
        value_path =osp.join(prefix,value_file_name)
        torch.save(self.policy.state_dict(), policy_path)
        torch.save(self.policy.state_dict(), value_path)