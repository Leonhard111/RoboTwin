# import packages and module here
import sys

import torch
import sapien.core as sapien
import traceback
import os
import numpy as np
from envs import *
from hydra import initialize, compose
from omegaconf import OmegaConf
from hydra.core.hydra_config import HydraConfig
from hydra import main as hydra_main
import pathlib
from omegaconf import OmegaConf

import yaml
from datetime import datetime
import importlib

from hydra import initialize, compose
from omegaconf import OmegaConf
from datetime import datetime

current_file_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(current_file_path)

sys.path.append(parent_directory)

from maniflow_model import *


def encode_obs(observation):  # Post-Process Observation
    obs = dict()
    # Normalize head camera to (C, H, W) float32 in range [0,1]
    head_cam = (np.moveaxis(observation["observation"]["head_camera"]["rgb"], -1, 0) / 255.0).astype(np.float32)
    obs['head_cam'] = head_cam

    # Normalize agent_pos: support nested dict, list, tuple or ndarray
    agent_pos = observation.get('joint_action', observation.get('agent_pos'))
    if isinstance(agent_pos, dict):
        # common key name is 'vector'
        agent_pos = agent_pos.get('vector', next(iter(agent_pos.values()), agent_pos))
    agent_pos = np.asarray(agent_pos, dtype=np.float32)
    obs['agent_pos'] = agent_pos

    # Normalize point cloud
    point_cloud = observation.get('pointcloud', observation.get('point_cloud'))
    point_cloud = np.asarray(point_cloud, dtype=np.float32)
    obs['point_cloud'] = point_cloud

    return obs


def get_model(usr_args):
    config_path = "./diffusion_policy/config"
    config_name = f"{usr_args['config_name']}.yaml"

    with initialize(config_path=config_path, version_base='1.2'):
        cfg = compose(config_name=config_name)

    # now = datetime.now()
    # run_dir = f"data/outputs/{now:%Y.%m.%d}/{now:%H.%M.%S}_{usr_args['config_name']}_{usr_args['task_name']}"

    task_name = usr_args['task_name']
    alg_name = usr_args['alg_name']
    addition_info = usr_args['addition_info']
    seed = usr_args['training_seed']
    exp_name = f"{task_name}-{alg_name}-{addition_info}"
    run_dir = os.path.join(parent_directory, "data", "outputs", exp_name + f"_seed{seed}")


    hydra_runtime_cfg = {
        "job": {
            "override_dirname": usr_args['task_name']
        },
        "run": {
            "dir": run_dir
        },
        "sweep": {
            "dir": run_dir,
            "subdir": "0"
        }
    }
    
    OmegaConf.set_struct(cfg, False)
    cfg.hydra = hydra_runtime_cfg
    cfg.task_name = usr_args["task_name"]
    cfg.expert_data_num = usr_args["expert_data_num"]
    cfg.raw_task_name = usr_args["task_name"]
    OmegaConf.set_struct(cfg, True)

    ManiFlow_Model = ManiFlow(cfg, usr_args, run_dir=run_dir)
    return ManiFlow_Model


def eval(TASK_ENV, model, observation):
    obs = encode_obs(observation)  # Post-Process Observation
    # instruction = TASK_ENV.get_instruction()

    if len(
            model.env_runner.obs
    ) == 0:  # Force an update of the observation at the first frame to avoid an empty observation window, `obs_cache` here can be modified
        model.update_obs(obs)

    start_time = time.time()
    actions = model.get_action()  # Get Action according to observation chunk
    end_time = time.time()
    latency = (end_time - start_time) * 1000  # 转换为毫秒
    print(f"***********推理延迟: {latency:.2f} ms***********")
    print(actions.shape)
      # Debugging breakpoint
    for action in actions:  # Execute each step of the action
        TASK_ENV.take_action(action)
        observation = TASK_ENV.get_obs()
        obs = encode_obs(observation)
        model.update_obs(obs)  # Update Observation, `update_obs` here can be modified


def reset_model(model):
    # Reset temporal aggregation state if enabled
    if model.temporal_agg:
        model.all_time_actions = torch.zeros([
            model.max_timesteps,
            model.max_timesteps + model.num_queries,
            model.state_dim,
        ]).to(model.device)
        model.t = 0
        print("Reset temporal aggregation state")
        # print(f"All time actions: {model.all_time_actions[:3, :3].cpu().numpy()}")  # Print first 10 actions for debugging
    else:
        model.t = 0

    # Reset observation cache
    model.env_runner.reset_obs()
