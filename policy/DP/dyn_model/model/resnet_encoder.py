import torch
import hydra
import dill
from einops import rearrange
from diffusion_policy.workspace.base_workspace import BaseWorkspace

class ResNetEncoder(torch.nn.Module):
    def __init__(self, policy_ckpt_path, view_names, ):
        super().__init__()
        self.policy_ckpt_path = policy_ckpt_path
        self.view_names = view_names
        self.emb_dim = 512
        self.latent_ndim = 2
        self.name = 'resnet'

        with open(self.policy_ckpt_path, 'rb') as f:
            payload = torch.load(f, pickle_module=dill)
            cfg = payload['cfg']
        
        # Plan C: Instantiate policy directly
        policy = hydra.utils.instantiate(cfg.policy)
        if cfg.training.use_ema and 'ema_model' in payload['state_dicts']:
            policy.load_state_dict(payload['state_dicts']['ema_model'])
        else:
            policy.load_state_dict(payload['state_dicts']['model'])

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        policy.to(device)
        policy.eval()

        self.obs_encoder = {}
        for view_name in self.view_names:
            if hasattr(policy.obs_encoder, 'obs_nets'):
                self.obs_encoder[view_name] = policy.obs_encoder.obs_nets[view_name].backbone
            elif hasattr(policy.obs_encoder, 'key_model_map'):
                if view_name in policy.obs_encoder.key_model_map:
                    self.obs_encoder[view_name] = policy.obs_encoder.key_model_map[view_name]
                else:
                    # Fallback lookup for mismatched keys (e.g. head_camera vs head_cam)
                    available_keys = list(policy.obs_encoder.key_model_map.keys())
                    found_key = None
                    # 1. Check known mappings
                    mappings = {
                        'head_camera': 'head_cam',
                        'wrist_camera': 'wrist_cam' # Potential other mismatch
                    }
                    if view_name in mappings and mappings[view_name] in available_keys:
                        found_key = mappings[view_name]
                    
                    # 2. Check for simple substring match if unique
                    if not found_key:
                        candidates = [k for k in available_keys if k in view_name or view_name in k]
                        if len(candidates) == 1:
                            found_key = candidates[0]
                    
                    if found_key:
                        print(f"Warning: View '{view_name}' not found directly. Mapping to available key '{found_key}'.")
                        self.obs_encoder[view_name] = policy.obs_encoder.key_model_map[found_key]
                    else:
                        raise KeyError(f"View {view_name} not found in policy.obs_encoder.key_model_map. Available keys: {available_keys}")
            else:
                raise AttributeError(f"policy.obs_encoder (type: {type(policy.obs_encoder)}) has neither 'obs_nets' nor 'key_model_map'")
        
        self.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))
        
        del policy
        torch.cuda.empty_cache()

    def forward(self, x):
        view_embs = {}
        for view_name in self.view_names:
            imgs = x[view_name]
            b = imgs.shape[0]
            imgs = rearrange(imgs, "b t ... -> (b t) ...")
            imgs_emb = self.obs_encoder[view_name](imgs)
            
            # Robust pooling logic for various backbone outputs
            if imgs_emb.ndim == 4: # (B, C, H, W)
                imgs_emb = self.avgpool(imgs_emb) # -> (B, C, 1, 1)
                imgs_emb = imgs_emb.flatten(1)    # -> (B, C)
            elif imgs_emb.ndim == 3: # (B, N, D) - e.g. ViT tokens
                imgs_emb = imgs_emb.mean(dim=1)   # -> (B, D)
            elif imgs_emb.ndim == 2: # (B, D) - already pooled
                pass
            else:
                raise ValueError(f"View {view_name} output has unexpected shape: {imgs_emb.shape}")
                
            imgs_emb = imgs_emb.unsqueeze(1) # dummy patch dim
            imgs_emb = rearrange(imgs_emb, "(b t) p d -> b t p d", b=b)
            view_embs[view_name] = imgs_emb
        return view_embs


if __name__ == "__main__":
    ckpt = 'data/outputs/2025.03.26/04.49.31_train_diffusion_unet_hybrid_tool_hang_image_abs/checkpoints/240.ckpt'
    encoder = ResNetEncoder(ckpt, ['sideview', 'robot0_eye_in_hand'])
    bs = 2
    x_view_1 = torch.randn(bs, 2, 3, 128, 128).to('cuda')
    x_view_2 = torch.randn(bs, 2, 3, 128, 128).to('cuda')
    x = {'sideview': x_view_1, 'robot0_eye_in_hand': x_view_2}
    view_embs = encoder(x)
    for view_name, emb in view_embs.items():
        print(f"{view_name}: {emb.shape}")

