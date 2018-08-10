import argparse
from itertools import count
import logging
import numpy as np
import gym

import torch

from utils.args import get_args

from get_agent import get_agent

def main(args):

    model_store_sprefix = "snapshot"

    agent = get_agent(args)
    if args.resume:
        agent.load_model(model_store_sprefix)

    running_reward = None
    for t in count():
        reward = agent.step()
        running_reward = 0.9*running_reward + 0.1*reward if running_reward is not None else reward
        print("Episode:{}, running_Reward:{}".format(t,running_reward))
        print("Reward:{}".format(reward))
        if t % args.save_interval == 0:
            agent.snapshot(model_store_sprefix)

if __name__ == "__main__":
    args = get_args()
    main(args)
