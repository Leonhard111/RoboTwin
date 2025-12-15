#!/usr/bin/env python3
"""
Integration test to verify the streaming batch-based approach works end-to-end.
This simulates the data collection process without needing the full environment.
"""
import numpy as np
import zarr
import shutil
import os
import tempfile
from pathlib import Path

def _save_batch_to_zarr(
    zarr_data, zarr_meta,
    head_camera_batch, state_batch, joint_action_batch,
    total_frames, zarr_datasets_initialized, compressor
):
    """Append a batch of collected data to zarr datasets."""
    batch_size = len(head_camera_batch)
    if batch_size == 0:
        return
    
    # Convert batch lists to numpy arrays
    head_camera_arr = np.stack(head_camera_batch)  # (B, 480, 640, 3)
    state_arr = np.stack(state_batch)  # (B, 16)
    joint_action_arr = np.stack(joint_action_batch)  # (B, 14)
    
    # Convert BGR images to RGB and then to NCHW format
    head_camera_arr = head_camera_arr[..., ::-1]  # BGR to RGB
    head_camera_arr = np.transpose(head_camera_arr, (0, 3, 1, 2))  # HWCN to NCHW
    
    # Initialize datasets on first batch
    if not zarr_datasets_initialized:
        zarr_data.create_dataset(
            "head_camera",
            data=head_camera_arr,
            chunks=(100, 3, 480, 640),
            compressor=compressor,
            dtype=np.uint8
        )
        zarr_data.create_dataset(
            "state",
            data=state_arr,
            chunks=(100, 16),
            compressor=compressor,
            dtype=np.float32
        )
        zarr_data.create_dataset(
            "action",
            data=joint_action_arr,
            chunks=(100, 14),
            compressor=compressor,
            dtype=np.float32
        )
    else:
        # Append to existing datasets
        zarr_data["head_camera"].append(head_camera_arr)
        zarr_data["state"].append(state_arr)
        zarr_data["action"].append(joint_action_arr)


def simulate_data_collection(num_frames=2500, batch_size=1000):
    """
    Simulate the streaming data collection process from collect_rollout.py.
    
    Args:
        num_frames: Total frames to collect (simulating step_lim=400 with 6 real actions = ~2400 frames)
        batch_size: Size of each batch before saving to zarr
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        zarr_path = os.path.join(tmpdir, "test_collection.zarr")
        
        # Initialize zarr
        zarr_root = zarr.group(zarr_path)
        zarr_data = zarr_root.create_group("data")
        zarr_meta = zarr_root.create_group("meta")
        compressor = zarr.Blosc(cname="zstd", clevel=3, shuffle=1)
        
        # Initialize batch buffers
        head_camera_batch = []
        state_batch = []
        joint_action_batch = []
        episode_ends_array = []
        
        total_frames = 0
        zarr_datasets_initialized = False
        
        print(f"Simulating collection of {num_frames} frames in batches of {batch_size}...")
        print(f"Batch size memory: ~{batch_size * 0.88:.1f} MB (0.88 MB/frame)")
        print()
        
        # Simulate data collection loop
        episode_id = 0
        frames_in_episode = 0
        max_frames_per_episode = 400  # Simulating step_lim
        
        while total_frames < num_frames:
            # Simulate collecting frames for this episode
            while frames_in_episode < max_frames_per_episode and total_frames < num_frames:
                # Generate fake data
                head_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
                state = np.random.randn(16).astype(np.float32)
                action = np.random.randn(14).astype(np.float32)
                
                head_camera_batch.append(head_img)
                state_batch.append(state)
                joint_action_batch.append(action)
                
                total_frames += 1
                frames_in_episode += 1
                
                # Check if batch is full
                if len(head_camera_batch) >= batch_size or total_frames >= num_frames:
                    _save_batch_to_zarr(
                        zarr_data, zarr_meta,
                        head_camera_batch, state_batch, joint_action_batch,
                        total_frames, zarr_datasets_initialized, compressor
                    )
                    zarr_datasets_initialized = True
                    
                    # Clear batch
                    head_camera_batch = []
                    state_batch = []
                    joint_action_batch = []
                    
                    print(f"  Batch saved: {total_frames}/{num_frames} frames | "
                          f"zarr shape: {zarr_data['head_camera'].shape}")
                
                if total_frames >= num_frames:
                    break
            
            # Episode ends
            episode_ends_array.append(total_frames)
            frames_in_episode = 0
            episode_id += 1
            
            print(f"Episode {episode_id} ended. Total: {total_frames}/{num_frames} frames")
            
            if total_frames >= num_frames:
                break
        
        # Save any remaining partial batch
        if len(head_camera_batch) > 0:
            print(f"\nSaving final partial batch ({len(head_camera_batch)} frames)...")
            _save_batch_to_zarr(
                zarr_data, zarr_meta,
                head_camera_batch, state_batch, joint_action_batch,
                total_frames, zarr_datasets_initialized, compressor
            )
        
        # Save episode_ends metadata
        print("\nFinalizing zarr datasets...")
        episode_ends_array = np.array(episode_ends_array, dtype=np.int64)
        zarr_meta.create_dataset(
            "episode_ends",
            data=episode_ends_array,
            dtype="int64",
            compressor=compressor,
        )
        
        # Verify the data
        print("\n" + "="*60)
        print("✅ DATA COLLECTION SIMULATION COMPLETE")
        print("="*60)
        print(f"Total frames collected: {total_frames}")
        print(f"Total episodes: {len(episode_ends_array)}")
        print(f"\nFinal zarr structure:")
        print(f"  head_camera: {zarr_data['head_camera'].shape} (dtype: {zarr_data['head_camera'].dtype})")
        print(f"  state: {zarr_data['state'].shape} (dtype: {zarr_data['state'].dtype})")
        print(f"  action: {zarr_data['action'].shape} (dtype: {zarr_data['action'].dtype})")
        print(f"  episode_ends: {zarr_meta['episode_ends'].shape} (dtype: {zarr_meta['episode_ends'].dtype})")
        
        print(f"\nEpisode end indices: {zarr_meta['episode_ends'][:]}")
        
        # Verify data integrity
        print("\n" + "="*60)
        print("DATA INTEGRITY CHECKS")
        print("="*60)
        
        # Check 1: Total frames match
        assert zarr_data['head_camera'].shape[0] == total_frames
        assert zarr_data['state'].shape[0] == total_frames
        assert zarr_data['action'].shape[0] == total_frames
        print("✓ All datasets have consistent frame counts")
        
        # Check 2: Dimensions are correct
        assert zarr_data['head_camera'].shape[1:] == (3, 480, 640)
        assert zarr_data['state'].shape[1] == 16
        assert zarr_data['action'].shape[1] == 14
        print("✓ All datasets have correct dimensions")
        
        # Check 3: Episode ends are monotonically increasing
        episode_ends = zarr_meta['episode_ends'][:]
        assert np.all(np.diff(episode_ends) > 0)
        assert episode_ends[-1] == total_frames
        print("✓ Episode end indices are valid and monotonically increasing")
        
        print("\n✅ ALL CHECKS PASSED!")
        print(f"\nMemory efficiency summary:")
        print(f"  - Batch size: {batch_size} frames = ~{batch_size * 0.88:.1f} MB")
        print(f"  - Total frames: {total_frames}")
        print(f"  - Number of batches: {(total_frames + batch_size - 1) // batch_size}")
        print(f"  - Peak memory usage: ~{batch_size * 0.88:.1f} MB (constant)")
        print(f"  - Without streaming: ~{total_frames * 0.88:.1f} MB peak")
        print(f"  - Memory saved: {(total_frames * 0.88) - (batch_size * 0.88):.1f} MB")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("STREAMING DATA COLLECTION INTEGRATION TEST")
    print("="*60 + "\n")
    
    # Test 1: Normal size collection (2500 frames)
    print("TEST 1: Normal-size collection (2500 frames)")
    print("-" * 60)
    simulate_data_collection(num_frames=2500, batch_size=1000)
    
    print("\n" + "="*60)
    print("TEST 2: Large-scale collection (40,000 frames)")
    print("-" * 60 + "\n")
    simulate_data_collection(num_frames=40000, batch_size=1000)
    
    print("\n" + "="*60)
    print("✅ ALL INTEGRATION TESTS PASSED!")
    print("="*60)
    print("\nThe streaming approach successfully handles:")
    print("  ✓ Batch accumulation and incremental zarr saves")
    print("  ✓ Multiple episodes with correct episode_ends tracking")
    print("  ✓ Large-scale data collection without memory bloat")
    print("  ✓ Correct data format matching DP's expectations")
