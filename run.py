import argparse
import logging
import numpy as np
import gym
from itertools import count

import torch
import torch.optim as optim

from utils.args import get_args

from get_agent import get_agent

from models import *

from generator import ContinuousGenerator
from generator import DiscreteGenerator

from memory import Memory

from agent import PPOAgent
from utils.wrapper import *

format = "%(asctime)s %(threadName)s %(levelname)s: %(message)s"
log_formatter = logging.Formatter(format)
logging.basicConfig(level=logging.INFO, format=format)



def prepro(I):
    """ prepro 210x160x3 into 6400 """
    I = I[35:195]
    I = I[::2, ::2, 0]
    I[I == 144] = 0
    I[I == 109] = 0
    I[I != 0 ] = 1
    return I.astype(np.float)

class StackFrame(gym.Wrapper):
    def __init__(self, env=None, history_length=4):
        super(StackFrame, self).__init__(env)
        self.history_length = history_length
        self.buffer = None

    def reset(self):
        state = prepro(self.env.reset())
        self.buffer = [state] * self.history_length
        return np.asarray(np.stack(self.buffer,axis=0))

    def step(self, action):
        state, reward, done, info = self.env.step(action)
        self.buffer.pop(0)
        self.buffer.append(prepro(state))
        return np.asarray(np.stack(self.buffer,axis=0)), reward, done, info


def main(args):

    model_store_sprefix = "snapshot"
    
    env = NormalizedEnv(gym.make(args.env))
    # env = StackFrame(gym.make(args.env))
# NormalizedEnv
    model = MLPContinuousActorCritic(env)
    
    # model = MLPDiscreteActorCritic(env)
    # model = TestConv( env.action_space.n )

    optimizer = optim.Adam( model.parameters(), lr=args.rllr )

    # cont_generator = ContinuousGenerator(args)
    dis_generator = DiscreteGenerator(args)

    memory = Memory(args)

    agent = PPOAgent(args, model, optimizer, env, dis_generator,memory
        , False)
    if args.resume:
        agent.load_model(model_store_sprefix)

    agent.train(model_store_sprefix, args.save_interval)

if __name__ == "__main__":
    args = get_args()
    main(args)
