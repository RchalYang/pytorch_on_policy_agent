import collections
import copy
import numpy as np
import os.path as osp
import gym
from itertools import count
import logging

import torch
import torch.nn as nn
from torch.distributions import Categorical
import torch.multiprocessing as mp

from utils.torch_utils import device
from utils.torch_utils import Tensor
import utils.math_utils as math_utils

from .base_agent import BaseAgent
from models import Policy
from models import Value



def sync_grad(model, shared_model):
    for param, shared_param in zip(model.parameters(), shared_model.parameters()):
        if shared_param.grad is not None:
            return
        shared_param.grad = param.grad


class A3CAgent(BaseAgent):
    def __init__(self,args, env_wrapper):

        super(A3CAgent,self).__init__(args, env_wrapper)

        self.policy = Policy(self.env.action_space.n)
        self.value = Value()

        self.algo="a3c"

    def _optimize(self, observations, actions, discounted_rewards):
        
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
        
    def step(self):

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

        return avg_rewards

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

    @staticmethod
    def actor_process(rank, args,shared_model_actor, shared_model_critic, counter, lock, env_wrapper, model_store_sprefix):
        
        logging.info("Worker {} started".format(rank))
        
        current_agent = A3CAgent(args, env_wrapper)
        
        logging.info("Worker {} agent created".format(rank))
        current_agent.policy.to(device)
        current_agent.value.to(device)

        logging.info("Worker {} model to device".format(rank))

        optimizer = torch.optim.Adam(shared_model_actor.parameters(), lr=args.rllr)
        value_optimizer = torch.optim.Adam(shared_model_critic.parameters(), lr = args.rllr)

        running_reward = None

        for i in range(args.max_iter):
            
            current_agent.policy.load_state_dict(shared_model_actor.state_dict())
            current_agent.value.load_state_dict(shared_model_critic.state_dict())

            logging.info("Worker {} agent synced model".format(rank))

            reward = current_agent.step()
            
            logging.info("Worker {} generated data".format(rank))

            optimizer.zero_grad()
            value_optimizer.zero_grad()
            logging.info("Worker {} zeroed grad".format(rank))

            sync_grad( current_agent.policy , shared_model_actor )
            sync_grad( current_agent.value  , shared_model_critic )
            
            logging.info("Worker {} synced grad".format(rank))

            optimizer.step()
            value_optimizer.step()

            running_reward = 0.9*running_reward + 0.1*reward if running_reward is not None else reward
        
            logging.info("Process:{}, Episode:{}, running_Reward:{}".format(rank, i,running_reward))
            logging.info("Process:{}, Reward:{}".format(rank, reward))

            with lock:
                if counter.value % args.save_interval == 0:
                    current_agent.snapshot(model_store_sprefix)
                counter.value += 1


    def train(self, model_store_sprefix, save_interval):
        process = []

        self.policy.share_memory()
        self.value.share_memory()

        counter = mp.Value('i', 0)
        lock = mp.Lock()

        logging.info("Try to Start Workers")

        for i in range(self.args.num_workers):
            p=mp.Process(target=self.actor_process, 
                        args=(i, self.args, self.policy, self.value, 
                        counter, lock, self.env_wrapper, model_store_sprefix)
                )
            p.start()
            process.append(p)

        for p in process:
            p.join()

        logging.info("Finished Training")