import argparse
import logging
import numpy as np
import gym
from itertools import count

import torch

from utils.args import get_args

from get_agent import get_agent


format = "%(asctime)s %(threadName)s %(levelname)s: %(message)s"
log_formatter = logging.Formatter(format)
logging.basicConfig(level=logging.INFO, format=format)

def main(args):

    model_store_sprefix = "snapshot"

    agent = get_agent(args)
    if args.resume:
        agent.load_model(model_store_sprefix)

    agent.train(model_store_sprefix, args.save_interval)

if __name__ == "__main__":
    args = get_args()
    main(args)
