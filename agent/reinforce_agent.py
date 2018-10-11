import collections
import copy
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.distributions import Normal

from utils.torch_utils import device, Tensor
import utils.math_utils as math_utils

from .base_agent import BaseAgent

class ReinforceAgent(BaseAgent):

    def __init__(self, args, model, optim, env, data_generator, memory, continuous):

        super(ReinforceAgent,self).__init__(args, model, optim, env, data_generator, memory, continuous)

        self.algo="reinforce"


    def step(self):

        for batch in self.memory.one_iteration():
            obs, acts, rews = batch
            self._optimize( obs, acts, rews )

        self.step_time += 1

    def _optimize(self, obs, acts, rews):
        
        self.optim.zero_grad()
        
        obs  = Tensor( obs )
        acts = Tensor( acts )
        rews = Tensor( rews ).unsqueeze(1)

        if self.continuous:
            mean, std = self.model( obs )
            
            dis = Normal(mean, std)
            
            log_prob = dis.log_prob( acts ).sum( -1, keepdim=True )

            ent = dis.entropy().sum( -1, keepdim=True )

        else:

            probs = self.model(obs)

            dis = F.softmax( probs, dim = 1 )
            dis = Categorical( dis )

            acts = acts.long()

            log_prob = dis.log_prob(acts)
            ent = dis.entropy()
        
        rews = ( rews - rews.mean()) / ( rews.std() + 1e-8)
        
        actor_loss = -log_prob * rews

        actor_loss = actor_loss.mean() - self.entropy_para * ent.mean()
        
        actor_loss.backward()

        self.optim.step()
