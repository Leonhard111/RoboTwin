import sys
import os
import pathlib
import importlib
import time
import numpy as np
import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import dill
import wandb
import json
from diffusion_policy.workspace.base_workspace import BaseWorkspace

sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

def encode_obs(observation):  # Post-Process Observation
    obs = dict()
    # Normalize head camera to (C, H, W) float32 in range [0,1]
    if "head_camera" in observation["observation"]:
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
    if point_cloud is not None:
        point_cloud = np.asarray(point_cloud, dtype=np.float32)
        obs['point_cloud'] = point_cloud

    return obs


@hydra.main(config_path="dyn_model/conf/planner", config_name="eval_transport")
def main(cfg: DictConfig):
    output_dir = cfg.output_dir

    if os.path.exists(output_dir):
        confirm = input(f"Output path {output_dir} already exists! Overwrite? (y/N): ")
        if confirm.lower() != 'y':
            sys.exit(1)
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    config_save_path = os.path.join(output_dir, 'eval_config.yaml')
    OmegaConf.save(config=cfg, f=config_save_path)
    print(f"Configuration saved to {config_save_path}")

    # Load policy_checkpoint
    with open(cfg.policy_checkpoint, 'rb') as f:
        payload = torch.load(f, pickle_module=dill)
    
    # Update configuration based on payload
    cfg_task_env_runner = payload['cfg']
    cfg_task_env_runner.n_action_steps = cfg.n_action_steps
    cfg_task_env_runner.task.env_runner.n_action_steps = cfg.n_action_steps
    cfg_task_env_runner.policy.n_action_steps = cfg.n_action_steps

    cfg_task_env_runner.task.env_runner.n_test = cfg.n_test
    cfg_task_env_runner.task.env_runner.n_test_vis = cfg.n_test
    cfg_task_env_runner.task.env_runner.n_train = 0
    cfg_task_env_runner.task.env_runner.n_train_vis = 0
    cfg_task_env_runner.task.env_runner.test_start_seed = cfg.test_start_seed

    if 'libero' in cfg.policy_checkpoint:
        cfg_task_env_runner.task.env_runner.dataset_path = cfg.dataset_path
        
    # Initialize workspace
    cls = hydra.utils.get_class(cfg_task_env_runner._target_)
    workspace = cls(cfg_task_env_runner, output_dir=output_dir)
    workspace: BaseWorkspace
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    
    # Get policy from workspace
    policy = workspace.model
    if cfg_task_env_runner.training.use_ema:
        policy = workspace.ema_model

    device = torch.device(cfg.device)
    policy.to(device)
    policy.eval()
    
    normalizer_dir = os.path.dirname(os.path.dirname(cfg.policy_checkpoint))
    normalizer_path = os.path.join(normalizer_dir, 'normalizer.pth')
    policy.normalizer.load_state_dict(torch.load(normalizer_path))
    policy.normalizer.to(device)

    policy.initialize_planner(
        planner_target=cfg.planner_target,
        demo_dataset_config=payload['cfg'].task.dataset,
        dynamics_model_ckpt=cfg.dynamics_model_checkpoint,
        action_step=cfg_task_env_runner.n_action_steps,
        output_dir=cfg.output_dir,
        guidance_start_timestep=cfg.guidance_start_timestep,
        guidance_scale=cfg.guidance_scale,
        threshold=cfg.threshold,
        demo_dataset_path=cfg.get('demo_dataset_path', None)
    )

    # Run evaluation - use env_runner_target from the planner config
    cfg_task_env_runner.task.env_runner._target_ = cfg.env_runner_target

    # Instantiate Env Runner
    if 'libero' in cfg.policy_checkpoint:
        cfg_task_env_runner.task.env_runner.dataset_path = cfg.dataset_path
        env_runner = hydra.utils.instantiate(
            cfg_task_env_runner.task.env_runner,
            output_dir=output_dir,
            task_dir=cfg_task_env_runner.task.env_runner.dataset_path
        )
    else:
        env_runner = hydra.utils.instantiate(
             cfg_task_env_runner.task.env_runner,
             output_dir=output_dir
        )

    policy.num_inference_steps = cfg.num_inference_steps

    # Setup RoboTwin Environment
    task_name = cfg.get("task_name", "beat_block_hammer")
    try:
        task_module = importlib.import_module(f"envs.{task_name}")
        TaskClass = getattr(task_module, task_name)
    except (ImportError, AttributeError):
        import envs
        TaskClass = getattr(envs, task_name)

    TASK_ENV = TaskClass()
    TASK_ENV.setup_demo()

    success_count = 0
    results = {}
    
    # Evaluation Loop
    for i in range(cfg.n_test):
        TASK_ENV.reset()
        env_runner.reset_obs()
        policy.reset()

        observation = TASK_ENV.get_obs()
        obs = encode_obs(observation)
        env_runner.update_obs(obs)

        done = False
        step = 0
        max_steps = cfg.n_action_steps * 5 # Heuristic max steps

        print(f"Episode {i+1}/{cfg.n_test} started")
        
        while not done and step < max_steps:
             # get action
             actions = env_runner.get_action(policy)
             
             # Execute actions
             for action in actions:
                 TASK_ENV.take_action(action)
                 observation = TASK_ENV.get_obs()
                 obs = encode_obs(observation)
                 env_runner.update_obs(obs)
                 
                 step += 1
                 # Optional: Check success condition if available in TASK_ENV
                 # if TASK_ENV.check_success():
                 #    done = True
                 #    success_count += 1
                 #    break
                 if step >= max_steps:
                     break
        
        print(f"Episode {i+1} finished with {step} steps")

    # Save evaluation results
    results['success_rate'] = success_count / cfg.n_test
    
    results_path = os.path.join(output_dir, 'eval_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, sort_keys=True)

    print(f"Evaluation results saved to {results_path}")

if __name__ == '__main__':
    main()
