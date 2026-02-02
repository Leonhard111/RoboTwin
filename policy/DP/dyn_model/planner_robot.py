import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from accelerate import Accelerator
from omegaconf import OmegaConf, open_dict
from pathlib import Path
from einops import rearrange
import copy
import numpy as np

from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.model.common.normalizer import LinearNormalizer
from dyn_model.plan import load_model
from dyn_model.datasets.robot_image_dset import RobotImageDynamicsModelDataset
from dyn_model.datasets.img_transforms import default_transform, get_eval_crop_transform, get_eval_crop_transform_resnet

class RobotPlanner:
    def __init__(
        self,
        demo_dataset_config,
        dynamics_model_ckpt,
        action_step=8,
        output_dir='debug/',
        demo_dataset_path=None
    ):
        self.accelerator = Accelerator()
        self.device = self.accelerator.device

        # demo dataset
        self.demo_dataset_config = demo_dataset_config
        self.demo_dataset_path = demo_dataset_path
        
        # wm
        dynamics_model_dir = os.path.dirname(os.path.dirname(dynamics_model_ckpt))
        hydra_cfg_path = os.path.join(dynamics_model_dir, "hydra.yaml")
        
        with open(hydra_cfg_path, "r") as f:
            model_cfg = OmegaConf.load(f)
        self.model_cfg = model_cfg

        self.dyn_model = load_model(Path(dynamics_model_ckpt), model_cfg, device=self.device)
        self.dyn_model = self.accelerator.prepare(self.dyn_model)
        self.dyn_model.eval()

        # normalizer
        wm_normalizer = LinearNormalizer()
        normalizer_path = os.path.join(dynamics_model_dir, "normalizer.pth")
        if os.path.exists(normalizer_path):
            wm_normalizer.load_state_dict(torch.load(normalizer_path, map_location=self.device))
        else:
            print(f"Warning: normalizer.pth not found at {normalizer_path}")
            
        self.dyn_model_normalizer = wm_normalizer.to(self.device)
        self.policy_action_normalizer = LinearNormalizer()

        # Config parameters
        self.use_crop = model_cfg.get('use_crop', False)
        self.original_img_size = model_cfg.get('original_img_size', 140)
        self.cropped_img_size = model_cfg.get('cropped_img_size', 128)
        self.view_names = model_cfg.view_names
        self.frameskip = model_cfg.get('frameskip', 1)
        self.action_step = action_step
        self.horizon = self.demo_dataset_config.horizon // 16 # Align with other planners

        # Load demo latents if dataset is available
        self.demo_latents = None
        self.get_demo_latents()
        
        self.timestep = 0
        self.output_dir = output_dir
        self.idx = 0

    def set_policy_action_normalizer(self, policy_action_normalizer):
        self.policy_action_normalizer = policy_action_normalizer
        self.policy_action_normalizer.to(self.device)

    def get_demo_latents(self):
        # Determine dataset path
        zarr_path = self.demo_dataset_path
        if not zarr_path and self.demo_dataset_config is not None:
             zarr_path = self.demo_dataset_config.get('zarr_path') or self.demo_dataset_config.get('train_data_path')
        
        if not zarr_path:
            print("No demo dataset path provided, skipping latent extraction.")
            return

        print(f"Loading demo dataset from {zarr_path}")
        
        # Use RobotImageDynamicsModelDataset
        dataset = RobotImageDynamicsModelDataset(
            zarr_path=zarr_path,
            num_hist=self.model_cfg.num_hist,
            num_pred=self.model_cfg.num_pred,
            frameskip=self.frameskip,
            view_names=self.view_names,
            use_crop=self.use_crop,
            train=False, # Eval mode
            original_img_size=self.original_img_size,
            cropped_img_size=self.cropped_img_size
        )
        
        dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)
        
        visual_latents = []
        proprio_latents = []
        action_latents = []
        
        print("Extracting latents...")
        with torch.no_grad():
            for batch in dataloader:
                batch = dict_apply(batch, lambda x: x.to(self.device, non_blocking=True))
                
                # Normalize using WM normalizer
                n_batch = self.dyn_model_normalizer.normalize(batch)
                n_obs = n_batch['obs']
                n_action = n_batch['action']
                
                # Unwrap model
                model = self.dyn_model.module if hasattr(self.dyn_model, 'module') else self.dyn_model
                
                # Encode
                vis_lat = model.encoder(n_obs)
                prop_lat = model.proprio_encoder(n_obs['state'])
                act_lat = model.action_encoder(n_action)
                
                visual_latents.append(vis_lat.cpu())
                proprio_latents.append(prop_lat.cpu())
                action_latents.append(act_lat.cpu())
        
        self.demo_latents = {
            'visual': torch.cat(visual_latents, dim=0),
            'proprio': torch.cat(proprio_latents, dim=0),
            'action': torch.cat(action_latents, dim=0)
        }
        print(f"Extracted demo latents: {self.demo_latents['visual'].shape}")

    def compute_current_reward(self, current_obs):
        # Implementation depends on reward type. 
        # For now, return 0 or placeholder as RoboTwin might not have a reward function defined in this context yet.
        return 0.0

    def compute_nn_reward(self, current_visual_latent, current_proprio_latent):
        if self.demo_latents is None:
            return 0.0
            
        # Example implementation using cosine similarity to demo latents (common in these pipelines)
        # This matches Logic in planner.py and planner_libero.py
        
        demo_vis = self.demo_latents['visual'].to(self.device)
        demo_prop = self.demo_latents['proprio'].to(self.device)
        
        # Simple nearest neighbor or kernel based reward
        # Using visual similarity
        sim_vis = torch.nn.functional.cosine_similarity(current_visual_latent, demo_vis, dim=-1)
        max_sim = torch.max(sim_vis)
        
        return max_sim.item()

    def compute_loss(self, sample, current_obs):
        # Used for evaluating model prediction against ground truth
        model = self.dyn_model.module if hasattr(self.dyn_model, 'module') else self.dyn_model
        
        # Prepare inputs
        sample = dict_apply(sample, lambda x: x.to(self.device, non_blocking=True))
        
        # Normalize
        n_sample = self.dyn_model_normalizer.normalize(sample)
        n_obs = n_sample['obs']
        n_action = n_sample['action']
        
        # Encode GT
        with torch.no_grad():
            gt_visual_latent = model.encoder(n_obs)
            gt_proprio_latent = model.proprio_encoder(n_obs['state'])
            gt_action_latent = model.action_encoder(n_action)
        
        # Predict
        # We need history. This method signature in original files assumes we extract history from sample/current_obs
        # But 'compute_loss' in planner.py seems to be doing something specific.
        # Let's align with the interface but maybe placeholder if not strictly used for planning yet.
        
        # Assuming sample has history structure:
        # For prediction, we usually need [t-H : t] to predict [t+1 : t+P]
        # Implementation details depend on VisualDynamicsModel forward pass.
        
        return {}


    def prepare_obs(self, current_obs, action_shape=None):
        # Transform observation dict from Env (numpy) to Model Input (Torch Tensor)
        
        # 1. Handle transforms (Resize/Crop)
        # Assuming current_obs contains images that might need transform
        
        # Need to construct a dummy dict to pass to get_eval_crop_transform if needed, 
        # or just apply manually.
        
        obs_dict = copy.deepcopy(current_obs)
        
        # Images
        for view in self.view_names:
            if view in obs_dict:
                img = obs_dict[view] # C,H,W or H,W,C
                
                # Check format. RobotTwin envs usually return CHW or HWC depending on config.
                # If HWC, move to CHW.
                if img.shape[-1] == 3: 
                    img = np.moveaxis(img, -1, 0)
                
                # Convert to tensor
                tensor_img = torch.from_numpy(img).float().to(self.device)
                
                # Add batch dimension if missing
                if tensor_img.ndim == 3:
                     tensor_img = tensor_img.unsqueeze(0)
                
                # Transforms
                if self.use_crop:
                    # Apply eval crop (center crop)
                    # We need to define transforms.
                    # Usually we scale 0-1 then crop/resize? 
                    # Default transform scales to 0-1 if not already.
                    # Let's assume input is 0-255 or 0-1.
                    pass 
                
                # Normalize 0-1 if in 0-255
                if tensor_img.max() > 1.0:
                    tensor_img = tensor_img / 255.0
                    
                obs_dict[view] = tensor_img

        # Proprio/State
        if 'state' in obs_dict:
            state = obs_dict['state']
            if isinstance(state, np.ndarray):
                state = torch.from_numpy(state).float().to(self.device)
            if state.ndim == 1:
                state = state.unsqueeze(0)
            obs_dict['state'] = state
            
        return obs_dict
