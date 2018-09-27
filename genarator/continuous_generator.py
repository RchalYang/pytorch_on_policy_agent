import gym
import torch
from torch.distributions import Normal
from data_generator import Generator

class ContinuousGenerator(Generator):
    def __init__(self, env, model, memory, param):
        self.model = model
        self.env = env
        self.memory = memory

        self.gae = param.gae
        self.max_episode_time_step = param.max_episode_time_step
        self.time_steps = param.time_step

    def generate_one_episode(self):
        """
        generate one episode data and save them on memory
        """
        episodes_reward = []
        curr_episodes = 0
        total_reward = 0
        
        observations, actions, rewards, values = [], [], [], []

        observation = self.env.reset()

        current_time_step = 0
        
        while current_time_step <= max_episode_time_step:

            observations.append(observation)

            with torch.no_grad():
                observation_tensor = Tensor(observation).unsqueeze(0)
                mean, log_std, std, value = self.model(observation_tensor)

            act_dis = Normal(probabilities)
            action = act_distribution.sample()
            actions.append(action.cpu().numpy())

            observation, reward, done, _ = self.env.step(action)
            rewards.append(reward)
            total_reward += reward

            if done:
                break
            
            current_time_step += 1

        if not done:
            observation_tensor = Tensor(observation).unsqueeze(0)
            with torch.no_grad():
                _, _, _, last_value = model( observation_tensor )
            last_value = last_value.item()

        advantages, estimate_returens = self.reward_processor(  rewards, values, last_value  )

        return observations, actions, advantages, estimate_returens, total_reward/self.episodes, current_time_step
