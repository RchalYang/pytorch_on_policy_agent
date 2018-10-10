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
        
        self.opt_times = args.opt_times


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

            ratio = torch.exp(log_prob - log_prob_old)            

        else:

            probs, values = self.model(obs)
            with torch.no_grad():
                probs_old, _ = self.model_old(obs)

            dis = F.softmax( probs, dim = 1 )
            dis_old = F.softmax(probs_old, dim = 1 )

            acts = acts.long()

            probs = dis.gather( 1, acts )
            probs_old = dis_old.gather( 1, acts )

            # dis = Categorical( probs )
            # dis_old = Categorical( probs_old )
            ratio = probs / ( probs_old + 1e-8 )            

            # log_prob     = dis.log_prob( acts ).unsqueeze(1)
            # log_prob_old = dis_old.log_prob( acts ).unsqueeze(1)

            ent = -( dis.log() * dis ).sum(-1)


        # Normalize the advantage
        advs = (advs - advs.mean()) / (advs.std() + 1e-8)
        surrogate_loss_pre_clip = ratio * advs
        surrogate_loss_clip = torch.clamp(ratio, 
                        1.0 - self.clip_para,
                        1.0 + self.clip_para) * advs
                        
        # print("ratio min:{} max:{}".format(ratio.detach().min().item(), ratio.detach().max().item()))

        surrogate_loss = -torch.mean(torch.min(surrogate_loss_clip, surrogate_loss_pre_clip))

        policy_loss = surrogate_loss - self.entropy_para * ent.mean()

        criterion = nn.MSELoss()
        critic_loss = criterion( values, est_rs )
        # print("Critic Loss:{}".format(critic_loss.item()))

        self.writer.add_scalar( "Training/Critic_Loss", critic_loss.item(), self.step_count )
        loss = policy_loss + self.value_loss_coeff * critic_loss

        loss.backward()

        self.optim.step()

        self.step_count += 1

    def step(self):
        """
        Executes an iteration of PPO
        """
        for i in range(self.opt_times):
            super(PPOAgent, self).step()
        
        self.model_old.load_state_dict( self.model.state_dict() )

