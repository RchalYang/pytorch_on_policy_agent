import torch
import numpy as np
import gym

from agent import ReinforceAgent
from agent import A2CAgent
from agent import TRPOAgent
from agent import PPOAgent
from agent import A3CAgent


import utils.torch_utils

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

def get_agent(args):
    
    if args.agent == "Reinforce":
        return ReinforceAgent(args, StackFrame)
    
    if args.agent == "A2C":
        return A2CAgent(args, StackFrame)
    
    if args.agent == "PPO":
        return PPOAgent(args, StackFrame, False)
    
    if args.agent == "TRPO":
        return TRPOAgent(args, StackFrame)    

    if args.agent == "A3C":
        utils.torch_utils.device = "cpu"
        return A3CAgent(args, StackFrame)