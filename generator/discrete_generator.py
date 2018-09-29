import gym

import torch
from torch.distributions import Categorical
import torch.nn.functional as F

from .data_generator import Generator
from utils.torch_utils import Tensor


class DiscreteGenerator(Generator):
    def __init__(self, param):
        super().__init__(param)

    def _generate_one_episode(self, env, model):
        """
        generate one episode data and save them on memory
        """
        total_reward = 0
        
        observations, actions, rewards, values = [], [], [], []

        observation = env.reset()

        current_time_step = 0
        
        while current_time_step <= self.max_episode_time_step:

            observations.append(observation)

            with torch.no_grad():
                observation_tensor = Tensor(observation).unsqueeze(0)
                probs, value = model(observation_tensor)

            probs = F.softmax(probs, dim = 1)
            act_dis = Categorical(probs)
            action = act_dis.sample()
            action = action.cpu().numpy()
            actions.append(action)
            
            observation, reward, done, _ = env.step(action[0])
            
            values.append(value.item())
            rewards.append(reward)
            total_reward += reward
            if done:
                break
            
            current_time_step += 1

        last_value = 0
        if not done:
            observation_tensor = Tensor(observation).unsqueeze(0)
            with torch.no_grad():
                _, last_value = model( observation_tensor )
            last_value = last_value.item()

        advantages, estimate_returens = self.reward_processor(  rewards, values, last_value  )
        
        return observations, actions, advantages, estimate_returens, total_reward, current_time_step
