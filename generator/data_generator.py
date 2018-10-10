import torch
import gym
import copy

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
        self.manager = mp.manager()
        
        self.counter = mp.Value('i', 0)
        self.lock = mp.Lock()

        self.model_dict = self.manager.Dict()

        # pool = mp.Pool( processes=param.num_process)
        self.semaphore = self.manager.Semaphore()

        self.shared_model = None
        self.process()

        self.num_process = param.num_process

        self.processes = []
        for i in range( self.num_process ):
            p = mp.Process( target = MP.generate_data_subprocess, args = ( dic,  ) ) )

    @staticmethod
    def traj_generator( model, env, horizon ):
        raise NotImplementedError
 
    @staticmethod    
    def subprocess (rank, dict, list, env, shared_model, horizon ):

        model = copy.deepcopy(shared_model)
        traj_gen = MPGenerator.traj_generator( model )

        while True:


    def generate_data(self, model, env, memory):

        for i
    


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
