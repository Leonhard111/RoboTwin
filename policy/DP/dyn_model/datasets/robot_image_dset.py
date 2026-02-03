import copy
import os
import numpy as np
import torch
import zarr
from filelock import FileLock
from torch.utils.data import Dataset
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.normalize_util import array_to_stats, get_range_normalizer_from_stat, get_image_range_normalizer, get_identity_normalizer_from_stat
from diffusion_policy.model.common.normalizer import LinearNormalizer
from dyn_model.datasets.img_transforms import default_transform, get_train_crop_transform_resnet, get_eval_crop_transform_resnet

class RobotImageDynamicsModelDataset(Dataset):
    def __init__(self, 
                 zarr_path, 
                 num_hist=1, 
                 num_pred=1, 
                 frameskip=1, 
                 view_names=['head_camera'], 
                 use_crop=False,
                 train=True,
                 original_img_size=None,
                 cropped_img_size=None,
                 action_dim=None):
        
        super().__init__()
        
        self.num_hist = num_hist
        self.num_pred = num_pred
        self.frameskip = frameskip
        self.num_frames = num_hist + num_pred
        self.use_crop = use_crop
        self.train = train
        self.view_names = view_names
        self.original_img_size = original_img_size
        self.cropped_img_size = cropped_img_size

        self.original_action_dim = action_dim
        # 1. Load data
        print(f"Loading data from {zarr_path}")
        self.replay_buffer = ReplayBuffer.copy_from_path(
            zarr_path,
            keys=view_names + ["state", "action"]
        )
        
        # 2. Extract data to memory
        # Visual Data
        self.imgs = {}
        for view_name in self.view_names:
            self.imgs[view_name] = self.replay_buffer[view_name]
            
        # Proprioception Data (map state -> proprio)
        self.states = self.replay_buffer['state']
        self.proprio_dim = self.states.shape[1]
        
        # Action Data
        self.actions = self.replay_buffer['action']
        self.action_dim = self.actions.shape[1]
        
        if action_dim is not None:
            assert self.action_dim == action_dim, f"Config action_dim {action_dim} != dataset {self.action_dim}"

        self.action_dim = self.original_action_dim * frameskip
        
        # 3. Compute valid anchor indices
        # TODO:检查下表是否从1开始
        self.episode_ends = self.replay_buffer.episode_ends[:]
        self.episode_start_indices = np.concatenate(([0], self.episode_ends[:-1]))
        self.episode_end_indices = self.episode_ends - 1
        
        self.valid_anchor_indices = []
        for start, end in zip(self.episode_start_indices, self.episode_end_indices):

            #感觉实际上应该等于action执行数
            sequence_duration = self.frameskip
            last_valid_start = end - sequence_duration
            
            if last_valid_start >= start:
                # Include the last valid point
                self.valid_anchor_indices.extend(range(start, last_valid_start))
                
        self.valid_anchor_indices = np.array(self.valid_anchor_indices)
        print(f"Dataset Loaded: {len(self.valid_anchor_indices)} valid sequences.")
        
        # 4. Transform
        self.transform = default_transform()
        # Note: If crop transforms are needed, they can be added here similar to RobomimicImageDynamicsModelDataset
        if self.use_crop:
            if self.train:
                self.transform = get_train_crop_transform_resnet(original_img_size, cropped_img_size)
            else:
                self.transform = get_eval_crop_transform_resnet(original_img_size, cropped_img_size)
        

    def __len__(self):
        return len(self.valid_anchor_indices)

    def __getitem__(self, idx):
        # Get start index
        start = self.valid_anchor_indices[idx]
        end = start + self.num_frames * self.frameskip
        obs_indices = list(range(start, end, self.frameskip))
        act_indices = list(range(start, end))
        act_indices[-self.frameskip:] = [obs_indices[-1] - 1] * self.frameskip
        obs = {}

        obs = {}
        obs['visual'] = {}
        for view_name in self.view_names:
            img_data = self.imgs[view_name][obs_indices]
            
            # Normalize and Channel First (N, H, W, C) -> (N, C, H, W)
            if img_data.shape[-1] == 3: 
                img_data = np.moveaxis(img_data, -1, 1)
                
            obs['visual'][view_name] = torch.from_numpy(img_data.astype(np.float32) / 255.0)

        # Proprio
        proprio_data = self.states[obs_indices]
        obs['proprio'] = torch.from_numpy(proprio_data.astype(np.float32))
        
        # Action
        act_data = self.actions[act_indices] 
        act = torch.from_numpy(act_data.astype(np.float32))

        # State (Target)
        state = torch.from_numpy(proprio_data.astype(np.float32))

        return tuple([obs, act, state])

    def get_normalizer(self, **kwargs) -> LinearNormalizer:
        normalizer = LinearNormalizer()
        
        # Action Normalizer
        act_stat = array_to_stats(self.actions)

        normalizer['act'] = get_range_normalizer_from_stat(act_stat)
        
        # State Normalizer
        state_stat = array_to_stats(self.states)
        normalizer['state'] = get_range_normalizer_from_stat(state_stat)
        
        # Visual Normalizer
        for view_name in self.view_names:
            normalizer[view_name] = get_image_range_normalizer()
            
        return normalizer
