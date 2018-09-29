import torch
import gym

class Generator:
    def __init__(self, param):

        self.gae = not param.no_gae

        if self.gae:
            self.reward_processor = self._generalized_advantage_estimation
        else:
            self.reward_processor = self._discount_reward

        self.max_episode_time_step = param.max_episode_time_step
        self.time_steps = param.time_steps

        self.tau = param.tau
        self.gamma = param.gamma

    def _generate_one_episode(self, env, model):
        """
        generate one episode data and save them on memory
        """
        raise NotImplementedError
    
    def generate_data(self, model, env, memory):
        
        memory.clear()

        current_timesteps = 0
        ave_epi_rew = 0
        epis = 0
        while current_timesteps < self.time_steps:
            obs, acts, advs, est_rs, epi_rew, epi_timesteps = self._generate_one_episode(env, model)
            memory.obs.extend( obs )
            memory.acts.extend( acts )
            memory.advs.extend( advs )
            memory.est_rs.extend( est_rs )

            ave_epi_rew = epi_rew / (epis+1) + ave_epi_rew * epis / ( epis + 1 )

            current_timesteps += epi_timesteps
            epis += 1
        
        return ave_epi_rew
    
    def _generalized_advantage_estimation(self, rewards, values, last_value):
        """
        use GAE to process rewards
        P.S: only one round no need to process done info
        """
        A = 0
        advantages = []
        estimate_returns = []

        values.append(last_value)

        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * values[t + 1] - values[t]
            A = delta + self.gamma * self.tau * A
            advantages.insert( 0, A )
            estimate_returns.insert( 0, A + values[t] )

        return advantages, estimate_returns

    def _discount_reward(self, rewards, values, last_value):
        """
        Compute the discounted reward to estimate return and advantages
        """
        advantages = []
        estimate_returns = []

        R = last_value
        for t in reversed(range(len(rewards))):
            R = rewards[t] + self.gamma * R
            advantages.insert( 0, R - values[t] )
            estimate_returns.insert( 0, R )

        return advantages, estimate_returns
