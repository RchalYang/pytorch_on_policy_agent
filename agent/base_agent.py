import collections
import copy
import numpy as np
import os
import stat
import os.path as osp
from itertools import count
import gym
import logging
import shutil

import torch
from torch.distributions import Categorical

from utils.torch_utils import device, Tensor
import utils.math_utils as math_utils

from tensorboardX import SummaryWriter

class BaseAgent:
    def __init__(self, args, model, optim, env, data_generator, memory, continuous):

        self.args = args

        self.env          = env
        self.model        = model
        self.optim        = optim

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

        work_dir = osp.join("log", args.id )
        if osp.exists( work_dir ):
            os.chmod(work_dir, stat.S_IWUSR)
            shutil.rmtree(work_dir)
        self.writer = SummaryWriter( work_dir )
        self.total_time_steps = 0

    def _soft_update_target(self, target, source):
        for t, s in zip(target.parameters(), source.parameters()):
            t.data.copy_( (1. - self.tau) * t.data + self.tau * s.data )

    def _optimize(self, observations, actions, rewards):
        raise NotImplementedError

    def step(self):
        raise NotImplementedError

    def load_model(self, prefix):
        model_file_name="{}_model_latest_model.pth".format(self.algo)
        model_path=osp.join(prefix, model_file_name)
        self.model.load_state_dict(torch.load( model_path))

    def snapshot(self, prefix):
        model_file_name="{}_model_latest_model.pth".format(self.algo)
        model_path=osp.join(prefix, model_file_name)
        torch.save(self.model.state_dict(), model_path)

    def train(self, model_store_sprefix, save_interval):

        logging.info(self.algo)

        running_reward = None
        
        for t in count():
            
            reward, batch_time_steps = self.data_generator.generate_data(self.model, self.env, self.memory)
            self.total_time_steps += batch_time_steps
            self.step()
            running_reward = 0.9*running_reward + 0.1*reward if running_reward is not None else reward
        
            print("Episode:{}, running_Reward:{}".format(t,running_reward))
            print("Reward:{}".format(reward))
            self.writer.add_scalar("Training/Reward", reward ,self.total_time_steps)
            if t % save_interval == 0:
                self.snapshot(model_store_sprefix)
