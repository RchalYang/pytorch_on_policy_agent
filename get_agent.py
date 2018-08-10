import torch
import numpy as np
import gym

from agent import ReinforceAgent
from agent import A2CAgent
from agent import TRPOAgent
from agent import PPOAgent

from models import Policy
from models import Value

def prepro(I):
    """ prepro 210x160x3 into 6400 """
    I = I[35:195]
    I = I[::2, ::2, 0]
    I[I == 144] = 0
    I[I == 109] = 0
    I[I != 0 ] = 1
    return I.astype(np.float)

class StackFrame(gym.Wrapper):
    def __init__(self, env=None, history_length=1):
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
    
    env = gym.make(args.env)
    env = StackFrame(env,4)

    policy = Policy(env.action_space.n)
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.rllr)

    value = Value()
    value_optimizer = torch.optim.Adam(value.parameters(), lr = args.rllr)

    if args.agent == "Reinforce":
        return ReinforceAgent(env,
            policy, optimizer,
            args.episodes, args.gamma,
            args.entropy_para, args.batch_size,
            args.tau
        )
    
    if args.agent == "A2C":
        return A2CAgent(env,
            policy, optimizer,
            value, value_optimizer,
            args.episodes, args.gamma,
            args.entropy_para, args.batch_size,
            args.tau
        )
    
    if args.agent == "PPO":
        return PPOAgent(env,
            policy, optimizer,
            value, value_optimizer,
            args.update_time, args.clip_para,
            args.episodes, args.gamma,
            args.entropy_para, args.batch_size,
            args.tau
        )
    
    if args.agent == "TRPO":
        return TRPOAgent(env,
            policy,
            value, value_optimizer,
            args.max_kl, args.cg_damping,
            args.cg_iters, args.residual_tol,
            args.episodes, args.gamma,
            args.entropy_para, args.batch_size,
            args.tau
        )    