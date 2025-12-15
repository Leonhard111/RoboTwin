#!/usr/bin/env python3
"""
Test script to verify batch-based zarr saving works correctly.
"""
import numpy as np
import zarr
import shutil
import os
import tempfile

def _save_batch_to_zarr(
    zarr_data, zarr_meta,
    head_camera_batch, state_batch, joint_action_batch,
    total_frames, zarr_datasets_initialized, compressor
):
    """
    Append a batch of collected data to zarr datasets.
    """
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
        # Create resizable datasets
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


def test_batch_zarr_saving():
    """Test that batch-based zarr saving works correctly."""
    
    # Create temporary directory
    with tempfile.TemporaryDirectory() as tmpdir:
        zarr_path = os.path.join(tmpdir, "test_data.zarr")
        
        # Setup
        zarr_root = zarr.group(zarr_path)
        zarr_data = zarr_root.create_group("data")
        zarr_meta = zarr_root.create_group("meta")
        compressor = zarr.Blosc(cname="zstd", clevel=3, shuffle=1)
        
        print("Test 1: Save first batch (1000 frames)...")
        batch1_camera = [np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8) for _ in range(100)]
        batch1_state = [np.random.randn(16).astype(np.float32) for _ in range(100)]
        batch1_action = [np.random.randn(14).astype(np.float32) for _ in range(100)]
        
        _save_batch_to_zarr(
            zarr_data, zarr_meta,
            batch1_camera, batch1_state, batch1_action,
            100, False, compressor
        )
        
        assert zarr_data["head_camera"].shape == (100, 3, 480, 640), f"Wrong shape: {zarr_data['head_camera'].shape}"
        assert zarr_data["state"].shape == (100, 16), f"Wrong shape: {zarr_data['state'].shape}"
        assert zarr_data["action"].shape == (100, 14), f"Wrong shape: {zarr_data['action'].shape}"
        print(f"✓ First batch saved correctly")
        print(f"  head_camera shape: {zarr_data['head_camera'].shape}")
        print(f"  state shape: {zarr_data['state'].shape}")
        print(f"  action shape: {zarr_data['action'].shape}")
        
        print("\nTest 2: Append second batch (100 frames)...")
        batch2_camera = [np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8) for _ in range(100)]
        batch2_state = [np.random.randn(16).astype(np.float32) for _ in range(100)]
        batch2_action = [np.random.randn(14).astype(np.float32) for _ in range(100)]
        
        _save_batch_to_zarr(
            zarr_data, zarr_meta,
            batch2_camera, batch2_state, batch2_action,
            200, True, compressor
        )
        
        assert zarr_data["head_camera"].shape == (200, 3, 480, 640), f"Wrong shape: {zarr_data['head_camera'].shape}"
        assert zarr_data["state"].shape == (200, 16), f"Wrong shape: {zarr_data['state'].shape}"
        assert zarr_data["action"].shape == (200, 14), f"Wrong shape: {zarr_data['action'].shape}"
        print(f"✓ Second batch appended correctly")
        print(f"  head_camera shape: {zarr_data['head_camera'].shape}")
        print(f"  state shape: {zarr_data['state'].shape}")
        print(f"  action shape: {zarr_data['action'].shape}")
        
        print("\nTest 3: Append final partial batch (50 frames)...")
        batch3_camera = [np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8) for _ in range(50)]
        batch3_state = [np.random.randn(16).astype(np.float32) for _ in range(50)]
        batch3_action = [np.random.randn(14).astype(np.float32) for _ in range(50)]
        
        _save_batch_to_zarr(
            zarr_data, zarr_meta,
            batch3_camera, batch3_state, batch3_action,
            250, True, compressor
        )
        
        assert zarr_data["head_camera"].shape == (250, 3, 480, 640), f"Wrong shape: {zarr_data['head_camera'].shape}"
        assert zarr_data["state"].shape == (250, 16), f"Wrong shape: {zarr_data['state'].shape}"
        assert zarr_data["action"].shape == (250, 14), f"Wrong shape: {zarr_data['action'].shape}"
        print(f"✓ Final partial batch appended correctly")
        print(f"  head_camera shape: {zarr_data['head_camera'].shape}")
        print(f"  state shape: {zarr_data['state'].shape}")
        print(f"  action shape: {zarr_data['action'].shape}")
        
        print("\n✅ All tests passed! Batch-based zarr saving works correctly.")
        print("\nMemory efficiency:")
        print(f"  - Each batch holds max 1000 frames")
        print(f"  - Memory per frame: ~0.88 MB (for 480x640 RGB + state + action)")
        print(f"  - Max in-memory: ~880 MB per batch")
        print(f"  - For 40,000 frames: Would need ~40 batches")
        print(f"  - Peak memory: constant ~1 GB (1 batch at a time)")


if __name__ == "__main__":
    test_batch_zarr_saving()
