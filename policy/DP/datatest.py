# from policy.DP.diffusion_policy.common.replay_buffer import ReplayBuffer
import os
from diffusion_policy.common.replay_buffer import ReplayBuffer
from pathlib import Path

p = Path(__file__).resolve().parent / 'data' / 'beat_block_hammer-demo_clean-50.zarr' 
rb = ReplayBuffer.create_from_path(str(p), mode='r')

# 总时间步数（所有 episode 拼起来的长度）
print(rb.n_steps)

# episode 数量
print(rb.n_episodes)

# 某个键的真实数组形状（第一维是时间步数）
print(rb.data['head_camera'].shape)

# 查看每个 episode 的结束索引（累积时间步数）
# print(rb.meta['episode_ends'][:])

# 查看 img 的 chunk 配置（告诉你按什么块切）
print(rb.get_chunks().get('head_camera'))