import collections
import copy
import numpy as np

import torch
import torch.nn as nn
from torch.distributions import Categorical

from utils.torch_utils import device, Tensor
import utils.math_utils as math_utils

from .base_agent import BaseAgent

class ReinforceAgent(BaseAgent):

    def __init__(self, args, model, optim, env, data_generator, memory, continuous):

        super(ReinforceAgent,self).__init__(args, model, optim, env, data_generator, memory, continuous)

        self.algo="reinforce"


    def _optimize(self, observations, actions, discounted_rewards):

        self.optimizer.zero_grad()
        
        observations = Tensor(observations)
        actions = Tensor(actions).long()
        discounted_rewards = Tensor(discounted_rewards)

        dis = self.policy(observations)
        dis = Categorical(dis)
        log_prob = dis.log_prob(actions)
        mean_entropy = dis.entropy().mean()

        discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-8)

        actor_loss = -log_prob * discounted_rewards

        actor_loss = actor_loss.mean() - self.entropy_para * mean_entropy
        
        actor_loss.backward()

        self.optimizer.step()
