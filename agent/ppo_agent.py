import collections
import copy
import numpy as np
import os.path as osp

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.distributions import Normal

from torch.nn.utils.convert_parameters import vector_to_parameters, parameters_to_vector

from utils.torch_utils import device, Tensor
import utils.math_utils as math_utils

from .a2c_agent import A2CAgent

class PPOAgent(A2CAgent):
    def __init__(self, args, model, optim, env, data_generator, memory, continuous):

        super(PPOAgent,self).__init__( args, model, optim, env, data_generator, memory, continuous)
        
        self.update_time  = args.update_time
        self.clip_para    = args.clip_para
        self.algo="ppo"


    def _optimize(self, obs, acts, advs, est_rs):

        self.optim.zero_grad()
        
        obs  = Tensor( obs )
        acts = Tensor( acts )
        advs = Tensor( advs ).unsqueeze(1)
        est_rs = Tensor( est_rs ).unsqueeze(1)

        if self.continuous:
            mean, std, values = self.model( obs )
            with torch.no_grad():
                mean_old, std_old, _ = self.model_old( obs )

            dis = Normal(mean, std)
            dis_old = Normal(mean_old, std_old)
            
            log_prob     = dis.log_prob( acts ).sum( -1, keepdim=True )
            log_prob_old = dis_old.log_prob( acts ).sum( -1, keepdim=True )

            ent = dis.entropy().sum( -1, keepdim=True )
            
        else:

            probs, values = self.model(obs)
            with torch.no_grad():
                probs_old, _ = self.model_old(obs)


            probs = F.softmax(probs, dim = 1)

            probs_old = F.softmax(probs_old, dim = 1)
            
            dis = Categorical( probs )
            dis_old = Categorical( probs_old )

            acts = acts.squeeze(1).long()

            log_prob     = dis.log_prob( acts ).unsqueeze(1)
            log_prob_old = dis_old.log_prob( acts ).unsqueeze(1)

            ent = dis.entropy()

        ratio = torch.exp(log_prob - log_prob_old)
        # Normalize the advantage
        advs = (advs - advs.mean()) / (advs.std() + 1e-8)
        surrogate_loss_pre_clip = ratio * advs
        surrogate_loss_clip = torch.clamp(ratio, 
                        1.0 - self.clip_para,
                        1.0 + self.clip_para) * advs
        print("ratio min:{} max:{}".format(ratio.detach().min().item(), ratio.detach().max().item()))
        surrogate_loss = -torch.mean(torch.min(surrogate_loss_clip, surrogate_loss_pre_clip))

        policy_loss = surrogate_loss - self.entropy_para * ent.mean()

        criterion = nn.MSELoss( )
        critic_loss = criterion(values, est_rs )
        print("Critic Loss:{}".format(critic_loss.item()))

        loss = policy_loss + self.value_loss_coeff * critic_loss

        loss.backward()

        self.optim.step()

    def step(self):
        """
        Executes an iteration of PPO
        """
        
        ave_epi_rew = super(PPOAgent, self).step()

        if self.step_time % self.update_time == 0:
            self.model_old.load_state_dict( self.model.state_dict() )
            # self._soft_update_target(self.value_old, self.value)
            print("Updated old model")

        return ave_epi_rew
