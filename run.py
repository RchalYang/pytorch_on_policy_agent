import argparse
import logging
import numpy as np
import gym
from itertools import count

import torch
import torch.optim as optim

from utils.args import get_args
from utils.wrapper import WarpFrame
from models import *

from generator import ContinuousGenerator
from generator import MPContinuousGenerator
from generator import DiscreteGenerator

from memory import Memory

from agent import PPOAgent
from agent import A2CAgent
from agent import TRPOAgent
from utils.wrapper import *

import tensorboardX

format = "%(asctime)s %(threadName)s %(levelname)s: %(message)s"
log_formatter = logging.Formatter(format)
logging.basicConfig(level=logging.INFO, format=format)

def get_functions(env, args):
    l = len(env.observation_space.shape)

    if l ==3 :
        env = WarpFrame(env)
        
    # env = NormalizedEnv( env )

    if isinstance( env.action_space, gym.spaces.Discrete ):
        if  l == 1:
            return env, DiscreteGenerator(args), MLPDiscreteActorCritic(env), False
        elif l ==3:
            return env, DiscreteGenerator(args), ConvDiscreteActorCritic(env), False
    
    if isinstance( env.action_space, gym.spaces.Box ):
        if l == 1:
            return env, ContinuousGenerator(args), MLPContinuousActorCritic(env), True
        elif l == 3:
            return env, ContinuousGenerator(args), ConvContinuousActorCritic(env), True
    
    raise Exception("Environment currently not supported")
    

def main(args):

    model_store_sprefix = "snapshot"
    
    # NormalizedEnv
    env = gym.make(args.env)

    env.seed(args.seed)
    torch.manual_seed(args.seed)

    env, generator, model, cont = get_functions(env, args) 
    
    optimizer = optim.Adam( model.parameters(), lr=args.rllr )

    memory = Memory(args)

    agent = PPOAgent(args, model, optimizer, env, generator,memory, cont)
    if args.resume:
        agent.load_model(model_store_sprefix)

    agent.train(model_store_sprefix, args.save_interval)

if __name__ == "__main__":
    args = get_args()
    main(args)
