import gym
from .wrapper import *

def make_env(args):

    env = gym.make( args.env )

    if args.frame_stack:
        env = FrameStack(env, args.frame_stack)
    
    if args.norm_env:
        env = NormalizedEnv(env)

    return env