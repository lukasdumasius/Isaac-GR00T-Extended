"""
Memory-Augmented Dataset Wrapper for Action Memory Training.

This wrapper applies sliding-window augmentation to generate multiple
training samples from each trajectory with incremental memory.
"""

import torch
import numpy as np
from torch.utils.data import Dataset
from typing import Dict, List, Optional
import sys
sys.path.append('/projects/bfxb/haorany7/MemFusionVLA/Isaac-GR00T-Extended')


class MemoryAugmentedDatasetWrapper(Dataset):
    """
    Wrapper that takes a base dataset and applies sliding-window augmentation.
    
    For each episode in the base dataset:
    - Extract the full trajectory (if available)
    - Generate multiple samples using sliding windows with stride
    - Each sample has its own memory (from past slices)
    
    This enables training cross-attention with realistic memory scenarios.
    """
    
    def __init__(
        self,
        base_dataset: Dataset,
        trajectory_encoder: torch.nn.Module,
        window_size: int = 16,
        action_memory_history_length: int = 512,
        stride: int = 4,
        max_samples_per_trajectory: Optional[int] = None,
        device: str = "cpu",
    ):
        """
        Args:
            base_dataset: Original dataset (e.g., LeRobotSingleDataset)
            trajectory_encoder: TrajectoryEncoder for encoding past slices
            window_size: Action prediction window size (should match action_horizon)
            stride: Offset stride for starting positions (e.g., 4 means 4 offsets: 0, 4, 8, 12)
                    Within each offset, windows are non-overlapping (jump by window_size)
            max_samples_per_trajectory: Limit samples per trajectory (for memory)
            device: Device to run trajectory_encoder on
        """
        self.base_dataset = base_dataset
        self.trajectory_encoder = trajectory_encoder.to(device).eval()
        self.window_size = window_size
        self.action_memory_history_length = action_memory_history_length
        self.stride = stride  # This is the offset stride, not window stride
        self.num_offsets = window_size // stride  # For window_size=16, stride=4 → 4 offsets
        self.max_samples_per_trajectory = max_samples_per_trajectory
        self.device = device
        
        # Pre-compute augmented sample indices
        # Format: (trajectory_id, offset, window_idx_within_offset)
        self.augmented_sample_map = []
        
        # Track sequences for sequential sampling
        # Each sequence is (trajectory_id, offset) and contains multiple windows
        self.sequences = {}  # sequence_id -> list of sample indices
        self.sample_to_sequence = {}  # sample_idx -> (sequence_id, position_in_sequence)
        
        print(f"[MemoryAugmentedDataset] Indexing dataset with {self.num_offsets}-offset augmentation")
        print(f"  Window size: {window_size}, Offset stride: {stride}")
        print(f"  Offsets: {list(range(0, window_size, stride))}")
        
        # Access trajectory information
        trajectory_ids = self.base_dataset.trajectory_ids
        trajectory_lengths = self.base_dataset.trajectory_lengths
        
        sequence_id = 0
        total_samples = 0
        
        skipped_trajectories = 0
        
        print(f"[MemoryAugmentedDataset] Total trajectories in base dataset: {len(trajectory_ids)}")
        print(f"[MemoryAugmentedDataset] Sample trajectory length: {trajectory_lengths[0] if len(trajectory_lengths) > 0 else 'N/A'}")

        for traj_idx, (traj_id, traj_len) in enumerate(zip(trajectory_ids, trajectory_lengths)):
            # Skip trajectories that are too short to contain even one window
            if traj_len < window_size:
                skipped_trajectories += 1
                continue

            # For each offset (0, 4, 8, 12)
            for offset in range(0, window_size, stride):
                if offset >= traj_len:
                    continue  # Skip if offset exceeds trajectory length
                
                # Start a new sequence for this (trajectory_id, offset) pair
                sequence_samples = []
                
                # From this offset, take non-overlapping windows
                window_idx = 0
                current_start = offset
                
                while current_start < traj_len:
                    # Add this window
                    sample_idx = len(self.augmented_sample_map)
                    self.augmented_sample_map.append((traj_id, offset, window_idx))
                    
                    # Track sequence membership
                    sequence_samples.append(sample_idx)
                    self.sample_to_sequence[sample_idx] = (sequence_id, window_idx)
                    
                    total_samples += 1
                    
                    # Move to next non-overlapping window
                    window_idx += 1
                    current_start = offset + window_idx * window_size
                    
                    # Check max_samples_per_trajectory limit
                    if self.max_samples_per_trajectory is not None and total_samples >= self.max_samples_per_trajectory * len(trajectory_ids):
                        break
                
                # Store this sequence
                if len(sequence_samples) > 0:
                    self.sequences[sequence_id] = sequence_samples
                    sequence_id += 1
                
                if self.max_samples_per_trajectory is not None and total_samples >= self.max_samples_per_trajectory * len(trajectory_ids):
                    break
            
            if self.max_samples_per_trajectory is not None and total_samples >= self.max_samples_per_trajectory * len(trajectory_ids):
                break
        
        print(f"[MemoryAugmentedDataset] Generated {len(self.augmented_sample_map)} samples from {len(trajectory_ids)} trajectories")
        print(f"[MemoryAugmentedDataset] Created {len(self.sequences)} memory sequences")
        if len(self.sequences) > 0:
            print(f"[MemoryAugmentedDataset] Average sequence length: {len(self.augmented_sample_map) / len(self.sequences):.2f}")
        else:
            print(f"[MemoryAugmentedDataset] WARNING: No valid sequences created!")
        if skipped_trajectories > 0:
            print(f"[MemoryAugmentedDataset] Skipped {skipped_trajectories} trajectories (too short: < {window_size} steps)")
    
    def __len__(self):
        return len(self.augmented_sample_map)
    
    def __getitem__(self, idx: int) -> Dict:
        """
        Returns a sample with target_actions.
        
        NOTE: Memory is NOT pre-computed here. It will be built dynamically
        in the collate function from previous samples in the same batch/sequence.
        
        Steps:
        1. Get trajectory_id, offset, window_idx from augmented_sample_map
        2. Calculate base_index = offset + window_idx * window_size
        3. Extract target window [base_index : base_index + window_size]
        4. Get vision/language data from base_index
        5. Add sequence metadata for collate function
        """
        trajectory_id, offset, window_idx = self.augmented_sample_map[idx]
        
        # Calculate the starting index for this window
        base_index = offset + window_idx * self.window_size
        
        # Get vision/language/state/action data from base_index
        # NOTE: base_dataset already handles action_horizon in its transforms
        # So we get action with shape (action_horizon, D_action) automatically
        step_data = self.base_dataset.get_step_data(trajectory_id, base_index)
        
        # Apply transforms to get the processed data
        # This will give us action with correct shape (action_horizon, D_action)
        step_data_transformed = self.base_dataset.transforms(step_data)

        # Build long-horizon action history for memory (CPU fast path via Arrow)
        action_history = self.base_dataset.get_action_history_efficient(
            trajectory_id=trajectory_id,
            end_index=base_index,
            history_length=self.action_memory_history_length,
        )
        # Concatenate action dims following the configured order
        action_history_concat = np.concatenate(
            [action_history[key] for key in self.base_dataset.modality_keys["action"]],
            axis=-1,
        ).astype(np.float32)
        step_data_transformed["action_history"] = action_history_concat

        # Build long-horizon state history for memory (match action history length)
        state_history = self.base_dataset.get_state_history_efficient(
            trajectory_id=trajectory_id,
            end_index=base_index,
            history_length=self.action_memory_history_length,
        )
        state_history_concat = np.concatenate(
            [state_history[key] for key in self.base_dataset.modality_keys["state"]],
            axis=-1,
        ).astype(np.float32)
        step_data_transformed["state_history"] = state_history_concat
        
        # Add sequence metadata (for collate function to build memory)
        sequence_id, position_in_sequence = self.sample_to_sequence[idx]
        step_data_transformed['sequence_id'] = sequence_id
        step_data_transformed['sequence_position'] = position_in_sequence
        step_data_transformed['sample_idx'] = idx
        
        return step_data_transformed

class SequentialMemorySampler:
    """
    Sampler that yields batches where each batch contains consecutive windows
    from the same memory sequence.
    
    This ensures that within a batch, samples can build upon each other's memory.
    """
    
    def __init__(self, dataset: MemoryAugmentedDatasetWrapper, batch_size: int, shuffle: bool = True, drop_last: bool = False):
        """
        Args:
            dataset: MemoryAugmentedDatasetWrapper with sequence information
            batch_size: Number of samples per batch
            shuffle: Whether to shuffle the order of sequences (not windows within sequence)
            drop_last: Whether to drop incomplete batches at the end of sequences
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        
        # Group samples by sequence
        self.sequences = dataset.sequences  # sequence_id -> list of sample indices
        self.sequence_ids = list(self.sequences.keys())
    
    def __iter__(self):
        # Shuffle sequence order (but not within sequences)
        if self.shuffle:
            import random
            sequence_order = self.sequence_ids.copy()
            random.shuffle(sequence_order)
        else:
            sequence_order = self.sequence_ids
        
        # Yield batches
        for seq_id in sequence_order:
            sample_indices = self.sequences[seq_id]
            
            # Split this sequence into batches
            for i in range(0, len(sample_indices), self.batch_size):
                batch = sample_indices[i:i + self.batch_size]
                
                # Drop last incomplete batch if requested
                if self.drop_last and len(batch) < self.batch_size:
                    continue
                
                yield batch
    
    def __len__(self):
        total_batches = 0
        for seq_id in self.sequence_ids:
            sample_indices = self.sequences[seq_id]
            num_batches = len(sample_indices) // self.batch_size
            if not self.drop_last and len(sample_indices) % self.batch_size > 0:
                num_batches += 1
            total_batches += num_batches
        return total_batches


def create_memory_collate_fn(eagle_processor, trajectory_encoder=None):
    """
    Create a custom collate function that builds memory dynamically within each batch.
    
    Key idea: Samples in a batch are from the same sequence in order.
    - Sample 0: memory = []
    - Sample 1: memory = encode(sample_0's actions)
    - Sample 2: memory = encode(sample_0, sample_1)
    - ...
    
    NOTE: trajectory_encoder is not used in this function anymore.
    Memory encoding will happen in GPU during model forward pass.
    
    Args:
        eagle_processor: EAGLE processor for standard collation
        trajectory_encoder: (Unused, kept for compatibility)
        
    Returns:
        Collate function that prepares metadata for GPU-based memory encoding
    """
    from gr00t.model.transforms import collate
    
    def dynamic_memory_collate(samples: List[Dict]) -> Dict:
        # First apply standard collation (handles vision, language, state, action)
        batch = collate(samples, eagle_processor)
        
        # Check if samples have sequence information
        if 'sequence_id' not in samples[0]:
            # Fallback: no memory
            batch['memory_keys'] = None
            batch['memory_values'] = None
            batch['memory_mask'] = None
            return batch
        
        # Verify all samples are from the same sequence (should be guaranteed by sampler)
        sequence_ids = [s['sequence_id'] for s in samples]
        if len(set(sequence_ids)) > 1:
            print(f"Warning: Batch contains samples from {len(set(sequence_ids))} different sequences. Memory may be incorrect.")
        
        # Verify samples are in order
        positions = [s['sequence_position'] for s in samples]
        if positions != sorted(positions):
            print(f"Warning: Samples are not in sequence order: {positions}")
        
        # Build memory metadata
        # Instead of encoding here (which is slow on CPU in worker process),
        # we'll just collect the action sequences and do encoding later in GPU
        batch_size = len(samples)
        
        # Stack action histories for GPU-side encoding
        if "action_history" in samples[0]:
            histories = [s["action_history"] for s in samples]
            batch["action_history"] = torch.as_tensor(np.stack(histories, axis=0), dtype=torch.float32)
        else:
            batch["action_history"] = None

        # Store information for dynamic memory building in main process
        batch['_memory_sequence_id'] = sequence_ids
        batch['_memory_positions'] = positions
        batch['_needs_memory_encoding'] = True  # Flag to indicate this batch needs memory processing
        
        return batch
    
    return dynamic_memory_collate


