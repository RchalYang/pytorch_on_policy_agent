import torch
import gym

class Generator:
    def __init__(self, env, model, memory, param):
        self.model = model
        self.env = env
        self.memory = memory

        self.gae = param.gae

        if self.gae:
            self.reward_processor = self._generalized_advantage_estimation
        else:
            self.reward_processor = self._discount_reward

        self.max_episode_time_step = param.max_episode_time_step
        self.time_steps = param.time_step

        self.tau = param.tau
        self.gamma = param.gamma

    def _generate_one_episode(self):
        """
        generate one episode data and save them on memory
        """
        raise NotImplementedError
    
    def generate_data(self):

        current_timesteps = 0
        while current_timesteps < self.time_steps:
            obs, acts, advs, est_rs, epi_rew, epi_timesteps = self.generate_one_episode()
            self.memory.obs.extend( obs )
            self.memory.acts.extend( acts )
            self.memory.advs.extend( advs )
            self.memory.est_rs.extend( ests )
            current_timesteps += epi_timesteps
    
    def _generalized_advantage_estimation(self, rewards, values, last_value):
        """
        use GAE to process rewards
        P.S: only one round no need to process done info
        """
        A = 0
        advantages = []
        estimate_return = []

        values.append(last_value)

        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * values[t + 1] - values[t].detach()
            A = delta + self.gamma * self.tau * A
            advantages.append( A )
            estimate_return.append( A + values[t] )

        return advantages, estimate_returns

    def _discount_reward(self, rewards, values, last_value):
        """
        Compute the discounted reward to estimate return and advantages
        """
        advantages = []
        estimate_return = []

        R = last_value
        for t in reversed(range(len(rewards))):
            R = rewards[t] + self.gamma * R
            advantages.insert( 0, R - values[t] )
            estimate_returns.insert( 0, R )

	    return advantages, estimate_returns
