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

format = "%(asctime)s %(threadName)s %(levelname)s: %(message)s"
log_formatter = logging.Formatter(format)
logging.basicConfig(level=logging.INFO, format=format)

def main(args):

    model_store_sprefix = "snapshot"
    
    env = gym.make(args.env)

    # model = MLPContinuousActorCritic(env)
    
    model = MLPDiscreteActorCritic(env)

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
