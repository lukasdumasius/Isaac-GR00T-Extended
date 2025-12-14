"""
Data augmentation strategy for action memory training.

From a single long trajectory, generate multiple training samples with
different memory states using sliding windows.
"""

import torch
from typing import List, Tuple, Dict


def augment_trajectory_with_memory(
    trajectory: torch.Tensor,
    window_size: int = 16,
    stride: int = 4,
    trajectory_encoder = None,
) -> List[Dict]:
    """
    From a single trajectory, generate multiple training samples with incremental memory.
    
    Args:
        trajectory: (T_total, D_action) - full trajectory
        window_size: Size of action prediction window (e.g., 16)
        stride: Stride for sliding window (e.g., 4)
        trajectory_encoder: TrajectoryEncoder instance
        
    Returns:
        List of dicts, each containing:
        {
            'target_actions': (window_size, D_action),
            'memory_latents': (N_memory, D_mem),  # N_memory varies per sample
        }
    """
    T_total, D_action = trajectory.shape
    
    if T_total < window_size:
        # Trajectory too short, return single sample with no memory
        # Pad to window_size
        pad_len = window_size - T_total
        padded = torch.cat([
            trajectory,
            trajectory[-1:].expand(pad_len, -1)  # Repeat last action
        ], dim=0)
        return [{
            'target_actions': padded,
            'memory_latents': torch.zeros(0, trajectory_encoder.output_dim) if trajectory_encoder else None,
        }]
    
    samples = []
    
    # Generate sliding windows
    for start_idx in range(0, T_total - window_size + 1, stride):
        end_idx = start_idx + window_size
        
        # Target actions for this sample
        target_actions = trajectory[start_idx:end_idx]  # (window_size, D_action)
        
        # Build memory from all PAST slices
        memory_latents = []
        
        if start_idx > 0:
            # Encode all past trajectory slices
            for past_start in range(0, start_idx, stride):
                past_end = min(past_start + window_size, start_idx)
                past_slice = trajectory[past_start:past_end]  # (<=window_size, D_action)
                
                # Pad if needed
                if past_slice.shape[0] < window_size:
                    pad_len = window_size - past_slice.shape[0]
                    past_slice = torch.cat([
                        past_slice,
                        past_slice[-1:].expand(pad_len, -1)
                    ], dim=0)
                
                # Encode
                if trajectory_encoder is not None:
                    with torch.no_grad():
                        latent = trajectory_encoder(past_slice.unsqueeze(0))  # (1, 1, D_mem)
                        memory_latents.append(latent.squeeze(0))  # (1, D_mem)
        
        # Stack memory
        if len(memory_latents) > 0:
            memory_tensor = torch.stack(memory_latents, dim=0)  # (N_memory, D_mem)
        else:
            # No memory for first sample
            memory_tensor = torch.zeros(0, memory_latents[0].shape[-1] if memory_latents else 512)
        
        samples.append({
            'target_actions': target_actions,
            'memory_latents': memory_tensor,
        })
    
    return samples


def collate_memory_augmented_batch(
    samples: List[Dict],
    max_memory_size: int = 100,
) -> Dict[str, torch.Tensor]:
    """
    Collate samples with variable-length memory into a batch.
    
    Args:
        samples: List of dicts with 'target_actions' and 'memory_latents'
        max_memory_size: Maximum memory size to pad to
        
    Returns:
        {
            'target_actions': (B, window_size, D_action),
            'memory_keys': (B, max_N, D_mem),
            'memory_values': (B, max_N, D_mem),
            'memory_mask': (B, max_N),  # 1 for valid, 0 for padding
        }
    """
    batch_size = len(samples)
    
    # Get dimensions
    window_size = samples[0]['target_actions'].shape[0]
    D_action = samples[0]['target_actions'].shape[1]
    D_mem = samples[0]['memory_latents'].shape[1] if samples[0]['memory_latents'].numel() > 0 else 512
    
    # Find max memory length in this batch
    max_memory_len = max(s['memory_latents'].shape[0] for s in samples)
    max_memory_len = min(max_memory_len, max_memory_size)  # Cap at max_memory_size
    
    # Initialize tensors
    target_actions = []
    memory_keys_list = []
    memory_values_list = []
    memory_masks = []
    
    for sample in samples:
        target_actions.append(sample['target_actions'])
        
        mem_latents = sample['memory_latents']  # (N_memory, D_mem)
        N = mem_latents.shape[0]
        
        # Truncate if too long
        if N > max_memory_size:
            mem_latents = mem_latents[-max_memory_size:]  # Keep most recent
            N = max_memory_size
        
        # Pad if needed
        if N < max_memory_len:
            pad_len = max_memory_len - N
            pad = torch.zeros(pad_len, D_mem, device=mem_latents.device)
            mem_latents_padded = torch.cat([mem_latents, pad], dim=0)
            
            # Mask: 1 for valid, 0 for padding
            mask = torch.cat([
                torch.ones(N),
                torch.zeros(pad_len)
            ], dim=0)
        else:
            mem_latents_padded = mem_latents
            mask = torch.ones(N)
        
        memory_keys_list.append(mem_latents_padded)
        memory_values_list.append(mem_latents_padded)
        memory_masks.append(mask)
    
    # Stack into batch
    batch = {
        'target_actions': torch.stack(target_actions),  # (B, window_size, D_action)
        'memory_keys': torch.stack(memory_keys_list),  # (B, max_N, D_mem)
        'memory_values': torch.stack(memory_values_list),  # (B, max_N, D_mem)
        'memory_mask': torch.stack(memory_masks),  # (B, max_N)
    }
    
    return batch


# Example usage in dataset
"""
class MemoryAugmentedDataset(Dataset):
    def __init__(self, base_dataset, trajectory_encoder, window_size=16, stride=4):
        self.base_dataset = base_dataset
        self.trajectory_encoder = trajectory_encoder
        self.window_size = window_size
        self.stride = stride
        
        # Pre-compute augmented samples
        self.augmented_samples = []
        for episode_idx in range(len(base_dataset)):
            episode = base_dataset[episode_idx]
            full_trajectory = episode['full_trajectory']  # Need this field
            
            aug_samples = augment_trajectory_with_memory(
                full_trajectory,
                window_size=window_size,
                stride=stride,
                trajectory_encoder=trajectory_encoder,
            )
            
            for aug_sample in aug_samples:
                self.augmented_samples.append({
                    **episode,  # Keep vision, text, etc.
                    'action': aug_sample['target_actions'],
                    'memory_latents': aug_sample['memory_latents'],
                })
    
    def __len__(self):
        return len(self.augmented_samples)
    
    def __getitem__(self, idx):
        return self.augmented_samples[idx]
"""

