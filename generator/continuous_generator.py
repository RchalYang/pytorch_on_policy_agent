import gym

import torch
from torch.distributions import Normal
import torch.multiprocessing as mp

from .data_generator import Generator

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

# class MPContinuousGenerator(Generator):
#     def __init__(self, param):
#         super().__init__(param)
#         manager = mp.manager()
        
#         counter = mp.Value('i', 0)
#         lock = mp.Lock()

#         pool = mp.Pool( processes=param.num_process)

#         self.shared_model = None

#     @staticmethod    
#     def generate_data_single_process(model, env, model, horizon ):

#         mirror = mirror_id is not None
#         t = 0
#         ac = env.action_space.sample() # not used, just so we have the datatype
#         new = True # marks if we're on first timestep of an episode
#         ob = env.reset()

#         cur_ep_ret = 0 # return in current episode
#         cur_ep_ret_all = {}
#         cur_ep_len = 0 # len of current episode
#         ep_rets = [] # returns of completed episodes in this segment
#         ep_rets_all = {}
#         ep_lens = [] # lengths of ...

#         # Initialize history arrays
#         obs = np.array([ob for _ in range(horizon)])
#         rews = np.zeros(horizon, 'float32')
#         vpreds = np.zeros(horizon, 'float32')
#         news = np.zeros(horizon, 'int32')
#         acs = np.array([ac for _ in range(horizon)])
#         prevacs = acs.copy()
#         if mirror:
#             mirror_obs = obs.copy()
#             mirror_acs = acs.copy()

#         while True:
#             prevac = ac
#             ac, vpred = pi.act(stochastic, np.array(ob))
#             if mirror:
#                 mirror_ob = ob[mirror_id[0]]
#                 if len(mirror_id)>2:
#                     mirror_ob *= mirror_id[2]
#                 mirror_ac, _ = pi.act(stochastic, np.array(mirror_ob))
#                 mirror_ac = mirror_ac[mirror_id[1]]
#             # Slight weirdness here because we need value function at time T
#             # before returning segment [0, T-1] so we get the correct
#             # terminal value
#             if t > 0 and t % horizon == 0:
#                 seg_dict = {"ob" : obs, "rew" : rews, "vpred" : vpreds, "new" : news,
#                             "ac" : acs, "prevac" : prevacs, "nextvpred": vpred * (1 - new),
#                             "ep_rets" : ep_rets, "ep_rets_all" : ep_rets_all, "ep_lens" : ep_lens}
#                 if mirror:
#                     seg_dict['mirror_ob'] = mirror_obs
#                     seg_dict['mirror_ac'] = mirror_acs
#                 yield seg_dict
#                 # Be careful!!! if you change the downstream algorithm to aggregate
#                 # several of these batches, then be sure to do a deepcopy
#                 ep_rets = []
#                 ep_rets_all = {}
#                 ep_lens = []
#             i = t % horizon
#             obs[i] = ob
#             vpreds[i] = vpred
#             news[i] = new
#             acs[i] = ac
#             prevacs[i] = prevac
#             if mirror:
#                 mirror_obs[i] = mirror_ob
#                 mirror_acs[i] = mirror_ac

#             rew = 0
#             rew_all = {}
#             for ai in range(action_repeat):
#                 ob, r, new, r_all = env.step(ac)
#                 rew = rew * ai / (ai + 1) + r / (ai + 1)
#                 if ai == 0:
#                     rew_all = r_all
#                 elif r_all:
#                     for name, val in r_all.items():
#                         rew_all[name] = rew_all[name] * ai / (ai + 1) + val / (ai + 1)
#                 if new:
#                     break
#             rews[i] = rew

#             cur_ep_ret += rew
#             if not cur_ep_ret_all:
#                 cur_ep_ret_all = rew_all
#             else:
#                 for name, val in rew_all.items():
#                     cur_ep_ret_all[name] += val

#             cur_ep_len += 1
#             if not ep_rets_all and cur_ep_ret_all:
#                 for name in cur_ep_ret_all.keys():
#                     ep_rets_all[name] = []
#             if new:
#                 ep_rets.append(cur_ep_ret)
#                 if ep_rets_all:
#                     for name, val in cur_ep_ret_all.items():
#                         ep_rets_all[name].append(val)
#                 ep_lens.append(cur_ep_len)
#                 cur_ep_ret = 0
#                 if cur_ep_ret_all:
#                     for name, _ in cur_ep_ret_all.items():
#                         cur_ep_ret_all[name] = 0
#                 cur_ep_len = 0
#                 ob = env.reset()
#             t += 1


#     def generate_data(self, model, env, memory):

#         # for i
#         pass
    


#     def _generate_one_episode(self, env, model):
#         """
#         generate one episode data and save them on memory
#         """
#         total_reward = 0
        
#         observations, actions, rewards, values = [], [], [], []

#         observation = env.reset()

#         current_time_step = 0
        
#         while current_time_step <= self.max_episode_time_step:

#             observations.append(observation)

#             with torch.no_grad():
#                 observation_tensor = Tensor(observation).unsqueeze(0)
#                 mean, std, value = model(observation_tensor)

#             act_dis = Normal(mean, std)
#             action = act_dis.sample()
#             action = action.squeeze(0).cpu().numpy()
#             actions.append(action)
            
#             observation, reward, done, _ = env.step(action)
#             # print(reward)
#             values.append(value.item())
#             rewards.append(reward)
#             total_reward += reward
#             if done:
#                 break
            
#             current_time_step += 1

#         last_value = 0
#         if not done:
#             observation_tensor = Tensor(observation).unsqueeze(0)
#             with torch.no_grad():
#                 _, _, last_value = model( observation_tensor )
#             last_value = last_value.item()

#         advantages, estimate_returens = self.reward_processor(  rewards, values, last_value  )

#         return observations, actions, advantages, estimate_returens, total_reward, current_time_step
