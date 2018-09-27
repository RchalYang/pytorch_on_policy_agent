import collections
import copy
import numpy as np
import os.path as osp
from itertools import count
import gym
import logging

import torch
from torch.distributions import Categorical

from utils.torch_utils import device, Tensor
import utils.math_utils as math_utils

class BaseAgent:
    def __init__(self, args, model, env, data_generator, memory, continuous):

        self.args = args
        self.env_wrapper = env_wrapper

        # self.env          = env_wrapper(gym.make(args.env))
        self.env          = env

        self.model = model

        self.data_generator = data_generator
        self.memory = memory

        self.gamma        = args.gamma
        self.episodes     = args.episodes
        self.entropy_para = args.entropy_para
        self.batch_size   = args.batch_size
        self.tau          = args.tau

        self.continuous = continuous

        self.step_time = 0
        self.algo="base"

    def _soft_update_target(self, target, source):
        for t, s in zip(target.parameters(), source.parameters()):
            t.data.copy_( (1. - self.tau) * t.data + self.tau * s.data )

    def _sample_episodes(self):
        """
        Return observation/action/rewards and average rewards for episodes

        """
        episodes_reward = []
        curr_episodes = 0
        total_reward = 0
        
        observations, actions = [], []

        while curr_episodes < self.episodes:

            curr_episodes += 1
            rewards = []
            observation = self.env.reset()

            while True:

                observations.append(observation)

                with torch.no_grad():
                    observation_tensor = Tensor(observation).unsqueeze(0)
                    probabilities = self.policy(observation_tensor)

                act_distribution = Categorical(probabilities)
                action = act_distribution.sample()
                actions.append(action)

                observation, reward, done, _ = self.env.step(action.item())
                rewards.append(reward)
                total_reward += reward

                if done:
                    episodes_reward.append(rewards)
                    break

        def _flatten(l): return [item for sublist in l for item in sublist]
        discounted_rewards = _flatten([math_utils.discount(epi, self.gamma) for epi in episodes_reward])

        return observations, actions, discounted_rewards, total_reward/self.episodes

    def _optimize(self, observations, actions, rewards):
        raise NotImplementedError

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
        policy_path=osp.join(prefix,policy_file_name)
        self.policy.load_state_dict(torch.load(policy_path))

    def snapshot(self, prefix):
        policy_file_name="{}_policy_latest_model.pth".format(self.algo)
        policy_path=osp.join(prefix,policy_file_name)
        torch.save(self.policy.state_dict(), policy_path)

    def train(self, model_store_sprefix, save_interval):

        logging.info(self.algo)

        running_reward = None
        for t in count():
            reward = self.step()
            running_reward = 0.9*running_reward + 0.1*reward if running_reward is not None else reward
        
            print("Episode:{}, running_Reward:{}".format(t,running_reward))
            print("Reward:{}".format(reward))
            if t % save_interval == 0:
                self.snapshot(model_store_sprefix)
