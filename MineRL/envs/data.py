import os
import shutil

import gym
import minerl
import numpy as np
from ray.rllib.evaluation.sample_batch_builder import SampleBatchBuilder
from ray.rllib.models.preprocessors import get_preprocessor
from ray.rllib.offline import InputReader
from ray.rllib.offline.json_writer import JsonWriter
from envs.wrappers import wrap
import numpy as np
from sklearn.neighbors import NearestNeighbors


class MineRLDataEnv(gym.Env):
    def __init__(self, config):
        super().__init__()

        data_dir = os.environ.get('MINERL_DATA_ROOT', 'data')

        num_workers = config.get('num_workers', 6)
        worker_batch_size = config.get('worker_batch_size', 1024)
        minimum_size_to_dequeue = config.get('minimum_size_to_dequeue', 32)
        force_download = config.get('force_download', False)
        minecraft_env = config.get("minecraft_env_name", "MineRLTreechopVectorObf-v0")

        self.data_loader = minerl.data.make(minecraft_env, data_dir, num_workers, worker_batch_size, minimum_size_to_dequeue, force_download)
        self.env_spec = self.data_loader.spec
        self.observation_space = self.env_spec.observation_space
        self.action_space = self.env_spec.action_space['vector']


        # Discrete action
        num_actions = config.get('num_actions', 32)
        kmeans_file = os.path.join(data_dir, f'{num_actions}-means', f'{self.env_spec.name}.npy')
        self.kmeans = np.load(kmeans_file)
        self.action_space = gym.spaces.Discrete(num_actions)
        self.nearest_neighbors = NearestNeighbors(n_neighbors=1).fit(self.kmeans)

        # Normalize observation:
        self._old_pov_space: gym.spaces.Box = self.env_spec.observation_space.spaces['pov']
        self._old_vec_space: gym.spaces.Box = self.env_spec.observation_space.spaces['vector']
        self._pov_space = gym.spaces.Box(0, 2, self._old_pov_space.low.shape, np.float32)
        self._vec_space = gym.spaces.Box(0, 2, self._old_vec_space.low.shape, np.float32)
        self.observation_space = gym.spaces.Dict(
            { 
                "pov":self._pov_space, 
                "vector":self._vec_space
            }
        )

        # Grayscale obs
        if config.get("gray_scale", True):
            assert isinstance(self.observation_space['pov'], gym.spaces.Box)
            low = np.min(self.observation_space['pov'].low, axis=2, keepdims=True)
            high = np.max(self.observation_space['pov'].high, axis=2, keepdims=True)
            pov_space = gym.spaces.Box(low, high, dtype=self.observation_space['pov'].dtype)
 
            self.observation_space = gym.spaces.Tuple((pov_space, self.observation_space['vector']))

        

        self.trajectory_names = self.data_loader.get_trajectory_names()
        self.index = 0

        # used by step() and reset()
        self.trajectory = None
        self.step_index = 0



    def step(self, action):
        prev_obs, action, reward, obs, done = self.trajectory[self.step_index]
        info = {
            'prev_obs': prev_obs,
            'action': action,
        }
        self.step_index += 1
        if self.step_index >= len(self.trajectory):
            if not done:
                print('Encountered end of trajectory when done returned False!')
            done = True

        return obs, reward, done, info

    def reset(self):
        self.trajectory = list(self.data_loader.load_data(self.trajectory_names[self.index]))
        self.index += 1
        self.index %= len(self.trajectory_names)
        self.step_index = 0
        obs, _, _, _, _ = self.trajectory[self.step_index]
        return obs

    def render(self, mode='human'):
        pass

    def action(self, action: int):
        return self.kmeans[action]

    def reverse_action(self, action: np.ndarray):
        action = self.env.reverse_action(action)
        action = np.reshape(action, (1, 64))
        distances, indices = self.nearest_neighbors.kneighbors(action)
        return int(indices[0].item())


def wrap_env(env: MineRLDataEnv, env_config):
    return wrap(env, **env_config)


def write_jsons(environment, data_dir, env_config, save_path, overwrite=False, fail_safe=True, **kwargs):
    env = MineRLDataEnv(env_config)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if len(os.listdir(save_path)) != 0:
        abs_save_path = os.path.abspath(save_path)
        if overwrite:
            print(f'Overwriting! {abs_save_path}')
            shutil.rmtree(abs_save_path)
        else:
            if fail_safe:
                print(f'Json data already exists at {abs_save_path}')
                return
            else:
                raise ValueError(f'Directory {abs_save_path} not empty!'
                                 f'Cannot overwrite existing data automatically, please delete old data if unused.')

    batch_builder = SampleBatchBuilder()
    writer = JsonWriter(save_path)
    prep = get_preprocessor(env.observation_space)(env.observation_space)

    for eps_id, trajectory_name in enumerate(env.trajectory_names):
        t = 0
        prev_action = None
        prev_reward = 0
        done = False
        try:
            obs = env.reset()
        except TypeError:
            continue
        while not done:
            new_obs, reward, done, info = env.step(env.action_space.sample())
            action = info['action']
            action = env.reverse_action(action)
            if prev_action is None:
                prev_action = np.zeros_like(action)
            batch_builder.add_values(
                t=t,
                eps_id=eps_id,
                agent_index=0,
                obs=prep.transform(obs),
                actions=action,
                action_prob=1.0,  # put the true action probability here
                rewards=reward,
                prev_actions=prev_action,
                prev_rewards=prev_reward,
                dones=done,
                infos={'trajectory_name': trajectory_name},
                new_obs=prep.transform(new_obs),
            )
            obs = new_obs
            prev_action = action
            prev_reward = reward
            t += 1
        writer.write(batch_builder.build_and_reset())


class MineRLReader(InputReader):
    def __init__(self, env_config, environment, data_dir='/workspaces/MineRL/MineRL/data', num_workers=6, worker_batch_size=4, minimum_size_to_dequeue=32):
        super().__init__()
        
        self.env_config = env_config
        self.env_config.update({'data_dir': data_dir})
        self.env_config.update({'data_pipeline': self.data_loader})

    def get_env(self, env):
        return wrap(env, **self.env_config)

    def next(self):
        self.data_pipeline.batch_iter()