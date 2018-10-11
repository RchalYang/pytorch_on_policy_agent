import torch
import gym
import copy
import multiprocessing as mp

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
        self.episodes = param.episodes

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
        for i in range(self.episodes):
            obs, acts, advs, est_rs, epi_rew, epi_timesteps = self._generate_one_episode(env, model)
            memory.obs.extend( obs )
            memory.acts.extend( acts )
            memory.advs.extend( advs )
            memory.est_rs.extend( est_rs )

            ave_epi_rew = epi_rew / (epis+1) + ave_epi_rew * epis / ( epis + 1 )

            current_timesteps += epi_timesteps
            epis += 1
        
        return ave_epi_rew, current_timesteps
    
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


class MPGenerator(Generator):
    def __init__(self, param):
        super().__init__(param)

        self.shared_model = None
        
        self.num_process = param.num_process

        self.pool = mp.Pool( processes=param.num_process)
    
    @staticmethod
    def _generate_one_episode(env, model, horizon, reward_processor ):
        """
        generate one episode data and save them on memory
        """
        raise NotImplementedError

    def __getstate__(self):
        self_dict = self.__dict__.copy()
        del self_dict['pool']
        return self_dict

    # def __setstate__(self, state):
    #     self.__dict__.update(state)

    def generate_data(self, model, env, memory):
        
        memory.clear()

        results = []
        for i in range(self.episodes):
            results.append( self.pool.apply_async( type(self)._generate_one_episode ,
                ( env, model, self.max_episode_time_step, self.reward_processor  )
              ))            

        current_timesteps = 0
        ave_epi_rew = 0
        epis = 0

        for i in range(self.episodes):
            result = results[i].get()
            obs, acts, advs, est_rs, epi_rew, epi_timesteps = result
            memory.obs.extend( obs )
            memory.acts.extend( acts )
            memory.advs.extend( advs )
            memory.est_rs.extend( est_rs )

            ave_epi_rew = epi_rew / (epis+1) + ave_epi_rew * epis / ( epis + 1 )

            current_timesteps += epi_timesteps
            epis += 1
        
        return ave_epi_rew, current_timesteps