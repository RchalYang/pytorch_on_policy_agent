import gym

import torch
from torch.distributions import Normal
import torch.multiprocessing as mp

from .data_generator import Generator
from .data_generator import MPGenerator

from utils.torch_utils import Tensor


class ContinuousGenerator(Generator):
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
                mean, std, value = model(observation_tensor)

            act_dis = Normal(mean, std)
            action = act_dis.sample()
            action = action.squeeze(0).cpu().numpy()
            actions.append(action)
            
            observation, reward, done, _ = env.step(action)
            # print(reward)
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
                _, _, last_value = model( observation_tensor )
            last_value = last_value.item()

        advantages, estimate_returens = self.reward_processor(  rewards, values, last_value  )

        return observations, actions, advantages, estimate_returens, total_reward, current_time_step

class MPContinuousGenerator(MPGenerator):

    @staticmethod
    def _generate_one_episode( env, model, horizon, reward_processor ):
        """
        generate one episode data and save them on memory
        """
        total_reward = 0
        
        observations, actions, rewards, values = [], [], [], []

        observation = env.reset()

        current_time_step = 0
        
        while current_time_step <= horizon:

            observations.append(observation)

            with torch.no_grad():
                observation_tensor = Tensor(observation).unsqueeze(0)
                mean, std, value = model(observation_tensor)

            act_dis = Normal(mean, std)
            action = act_dis.sample()
            action = action.squeeze(0).cpu().numpy()
            actions.append(action)
            
            observation, reward, done, _ = env.step(action)
            # print(reward)
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
                _, _, last_value = model( observation_tensor )
            last_value = last_value.item()

        advantages, estimate_returens = reward_processor(  rewards, values, last_value  )

        return (observations, actions, advantages, estimate_returens, total_reward, current_time_step )
