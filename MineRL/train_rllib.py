"""
Runs the training process through the RLLib train script
See help with:
$ python rllib_train.py --help
"""
import subprocess

import yaml
from ray.rllib.train import create_parser, run
import os
import ray
from ray import tune
import envs
from networks import CNNLSTM
from envs import MineRLDataEnv
import convert_to_json
from envs.data import write_jsons
from ray.rllib.models import ModelCatalog


def generate_kmeans(env):
    command = f'python /workspaces/MineRL/MineRL/generate_kmeans.py --env {env}'
    print('running:', command)
    subprocess.run(command.split())

def main():
    parser = create_parser()
    args = parser.parse_args()
    check_data(args)
    run(args, parser)

def json_converter(env_config):
    env_save_path = env_config.get("json_save_path", "/workspaces/MineRL/MineRL/data")
    print(f'saving jsons to {env_save_path}')

    env_name = env_config.get("minecraft_env_name", "MineRLTreechopVectorObf-v0")
    print(f'Writing data to json files for environment {env_name}')
    write_jsons(env_name, env_save_path, env_config, env_save_path, overwrite=False)


if __name__ == '__main__':
    from envs.data import write_jsons
    from envs.wrappers import wrap
    from envs.data import wrap_env
    env_config = {
                "data_dir": "/workspaces/MineRL/MineRL/data",
                "json_save_path": "/workspaces/MineRL/MineRL/data",
                "discrete": True,
                "action_repeat": 4,
                "num_stack": 1,
                "gray_scale": False,
                "minecraft_env_name": "MineRLTreechopVectorObf-v0",
                "overwrite": False,
                "num_actions": 32
    }
    generate_kmeans(env_config['minecraft_env_name'])
    json_converter(env_config)

    ModelCatalog.register_custom_model("CNNLSTM", CNNLSTM)
    tune.register_env('MineRLDataEnv', lambda env_config: MineRLDataEnv(env_config))
    ray.init(local_mode=True)
    tune.run(
        "DQN",
        checkpoint_freq=30,
        checkpoint_at_end=True,
        config={
            "env": 'MineRLDataEnv',
            "model": {
                "custom_model": "CNNLSTM",
            },
            "num_workers": 2,
            "num_envs_per_worker": 32,
            "num_cpus_per_worker": 5,
            "num_gpus_per_worker": 0.1,
            # uncomment for remote environments, not that fast
            # "remote_worker_envs": True,
            # "remote_env_batch_wait_ms": 1000,  # almost always wait for everyone to finish
            # trainer doesn't need all the gpu
            "num_gpus": 0.5,

            # Rainow args
            "double_q": False,
            "dueling": False,
            "num_atoms": 51,
            "noisy": True,
            "sigma0": 0.1, # noisy nets parameter
            "gamma": 0.99,
            "lr": .0001,
            "adam_epsilon": .00015,
            "grad_clip": 10,
            "hiddens": (512,),
            "learning_starts": 1600,  # 20000 -> 1600
            "buffer_size": 100000,
            "rollout_fragment_length": 1, # 4 -> 1
            "train_batch_size": 32,
            "exploration_config": {
                "epsilon_timesteps": 2,
                "final_epsilon": 0.0
            },
            "target_network_update_freq": 2000,
            "prioritized_replay": True,
            "prioritized_replay_alpha": 0.5,
            "prioritized_replay_beta": 0.4,
            "final_prioritized_replay_beta": 1.0,
            "prioritized_replay_beta_annealing_timesteps": 8000000, # Only 8mm training steps allowed
            "n_step": 20,  # 3 -> 20

            #rllib args
            "num_gpus": .5,

            # Torch
            "framework": "torch"
        }
    )