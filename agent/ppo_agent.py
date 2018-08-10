import collections
import copy
import numpy as np
import os.path as osp

import torch
import torch.nn as nn
from torch.distributions import Categorical
from torch.nn.utils.convert_parameters import vector_to_parameters, parameters_to_vector

from utils.torch_utils import device, Tensor
import utils.math_utils as math_utils

from .a2c_agent import A2CAgent

class PPOAgent(A2CAgent):
    def __init__(self,
                env,
                policy,
                policy_optimizer,
                value,
                value_optimizer,
                update_time,
                clip_para,
                episodes,
                gamma,
                entropy_para,
                batch_size,
                tau
                ):

        super(PPOAgent,self).__init__(env,
                policy,       policy_optimizer,
                value,        value_optimizer,
                episodes,     gamma,
                entropy_para, batch_size,
                tau)
        self.policy_old   = copy.deepcopy(policy)
        self.policy_old.to(device)
        
        self.update_time  = update_time
        self.clip_para    = clip_para
        self.algo="ppo"


    def _optimize(self, observations, actions, discounted_rewards):

        self.optimizer.zero_grad()
        
        observations = Tensor(observations)
        actions = Tensor(actions).long().unsqueeze(1)
        discounted_rewards = Tensor(discounted_rewards).unsqueeze(1)

        dis_old = self.policy_old(observations).detach()
        dis_new = self.policy(observations)

        prob_old = dis_old.gather(1, actions)
        prob_new = dis_new.gather(1, actions)

        ratio = prob_new / prob_old
        
        entropy = -(dis_new.log()*dis_new).sum(-1)
        mean_entropy = entropy.mean()

        baseline = self.value(observations).detach()
        advantage = discounted_rewards - baseline
        # Normalize the advantage
        advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

        surrogate_loss_pre_clip = ratio * advantage
        surrogate_loss_clip = torch.clamp(ratio, 
                        1.0 - self.clip_para,
                        1.0 + self.clip_para) * advantage
        print("ratio min:{} max:{}".format(ratio.detach().min().item(), ratio.detach().max().item()))
        surrogate_loss = -torch.mean(torch.min(surrogate_loss_clip, surrogate_loss_pre_clip))

        loss = surrogate_loss - self.entropy_para * mean_entropy
        loss.backward()

        self.optimizer.step()

        self.value_optimizer.zero_grad()
        values = self.value(observations)
        criterion = nn.MSELoss()
        critic_loss = criterion(values, discounted_rewards )
        critic_loss.backward()
        self.value_optimizer.step()
        print("MSELoss for Value Net:{}".format(critic_loss.item()))

    def step(self):
        """
        Executes an iteration of PPO
        """
        # Generate rollout
        observations, actions, discounted_rewards, avg_rewards = self._sample_episodes()

        total_num = len(actions)
        batch_num = (total_num // self.batch_size) if (total_num % self.batch_size == 0) \
            else  (total_num // self.batch_size + 1)

        for i in range(batch_num):
            batch_start = i     * self.batch_size
            batch_end   = (i+1) * self.batch_size

            batch_observations       = observations[batch_start : batch_end]
            batch_actions            = actions[batch_start : batch_end]
            batch_discounted_rewards = discounted_rewards[batch_start: batch_end]
            
            self._optimize(batch_observations, batch_actions, batch_discounted_rewards)
                    
        self.step_time += 1
        # print(self.step_time, self.update_time)
        if self.step_time % self.update_time == 0:
            self.policy_old.load_state_dict( self.policy.state_dict() )
            # self._soft_update_target(self.value_old, self.value)
            print("Updated old model")

        return avg_rewards
