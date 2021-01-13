# Shamelessly stolen from https://github.com/juliusfrost/minerl-rllib/blob/master/convert_data.py

import argparse
import json
import os

import yaml
from minerl.herobraine.envs import obfuscated_envs
from ray.tune.utils import merge_dicts

from envs.data import write_jsons

def json_converter(env_config):
    env_save_path = env_config.get("json_save_path", "/workspaces/MineRL/data/")
    print(f'saving jsons to {env_save_path}')

    for env_name in env_config.get("minecraft_env_name", "MineRLTreechopVectorObf-v0"):
        print(f'Writing data to json files for environment {env_name}')
        env_save_path = os.path.join(env_save_path, env_name)
        write_jsons(env_name, args.data_dir, env_config, env_save_path, overwrite=args.overwrite)

