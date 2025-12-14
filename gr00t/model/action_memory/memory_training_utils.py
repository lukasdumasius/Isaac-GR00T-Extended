"""
Example training wrapper showing how to integrate action memory into GR00T training.
Simplified version: Direct trajectory latent retrieval without LLM processing.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional
from transformers.feature_extraction_utils import BatchFeature

from gr00t.model.action_memory.memory_cross_attention import (
    TrajectoryEncoder,
    MemoryBank,
)


class ActionMemoryTrainingWrapper:
    """
    Wrapper to handle memory building and retrieval during training.
    
    Usage:
        wrapper = ActionMemoryTrainingWrapper(
            action_dim=7,
            memory_dim=512,
            trajectory_window=10,
            max_memory_capacity=100,
        )
        
        # In training loop:
        for batch in dataloader:
            memory_keys, memory_values = wrapper.build_per_sample_memory(
                batch_actions=batch['actions'],  # (B, T_total, D_action)
            )
            
            # Pass to model
            output = model(
                backbone_output=backbone_output,
                action_input=action_input,
                memory_keys=memory_keys,  # (B, N, D_memory)
                memory_values=memory_values,  # (B, N, D_memory)
            )
    """
    
    def __init__(
        self,
        action_dim: int,
        memory_dim: int = 512,
        trajectory_window: int = 10,
        max_memory_capacity: int = 100,
        encoder_type: str = "mlp",
        device: str = "cuda",
    ):
        self.action_dim = action_dim
        self.memory_dim = memory_dim
        self.trajectory_window = trajectory_window
        self.max_memory_capacity = max_memory_capacity
        self.device = device
        
        # Initialize trajectory encoder
        self.trajectory_encoder = TrajectoryEncoder(
            action_dim=action_dim,
            hidden_dim=memory_dim,
            output_dim=memory_dim,
            window_size=trajectory_window,
            encoder_type=encoder_type,
        ).to(device)
    
    def slice_trajectory(
        self, 
        actions: torch.Tensor, 
        window_size: int
    ) -> List[torch.Tensor]:
        """
        Slice a long trajectory into overlapping windows.
        
        Args:
            actions: (T, D_action)
            window_size: int
            
        Returns:
            List of trajectory slices, each (window_size, D_action)
        """
        T = actions.shape[0]
        slices = []
        
        # Create sliding windows
        for start_idx in range(0, T - window_size + 1, window_size // 2):  # 50% overlap
            end_idx = start_idx + window_size
            if end_idx <= T:
                slices.append(actions[start_idx:end_idx])
        
        # Handle remaining actions
        if len(slices) == 0 or slices[-1].shape[0] < window_size:
            # Pad last window if needed
            if T >= window_size:
                slices.append(actions[-window_size:])
        
        return slices
    
    def build_per_sample_memory(
        self,
        batch_actions: torch.Tensor,
    ) -> tuple:
        """
        Build per-sample memory banks for a batch.
        
        Args:
            batch_actions: (B, T_total, D_action) - full action trajectories for the batch
            
        Returns:
            memory_keys: (B, N_max, D_memory) - padded trajectory latents
            memory_values: (B, N_max, D_memory) - same as keys in simplified version
        """
        B, T_total, D_action = batch_actions.shape
        
        all_sample_keys = []
        all_sample_values = []
        max_memory_len = 0
        
        # Process each sample separately
        for sample_idx in range(B):
            sample_actions = batch_actions[sample_idx]  # (T_total, D_action)
            
            # Slice into trajectory windows
            trajectory_slices = self.slice_trajectory(sample_actions, self.trajectory_window)
            
            sample_keys = []
            sample_values = []
            
            # Encode each trajectory slice
            for traj_slice in trajectory_slices:
                # Add batch dimension and encode
                traj_slice_batch = traj_slice.unsqueeze(0)  # (1, window_size, D_action)
                
                with torch.no_grad() if not self.trajectory_encoder.training else torch.enable_grad():
                    traj_latent = self.trajectory_encoder(traj_slice_batch)  # (1, 1, D_memory)
                
                sample_keys.append(traj_latent)
                sample_values.append(traj_latent)  # Same as keys in simplified version
            
            # Stack keys and values for this sample
            if len(sample_keys) > 0:
                sample_keys_tensor = torch.cat(sample_keys, dim=1)  # (1, N, D_memory)
                sample_values_tensor = torch.cat(sample_values, dim=1)  # (1, N, D_memory)
            else:
                # Empty memory
                sample_keys_tensor = torch.zeros(1, 0, self.memory_dim, device=self.device)
                sample_values_tensor = torch.zeros(1, 0, self.memory_dim, device=self.device)
            
            all_sample_keys.append(sample_keys_tensor)
            all_sample_values.append(sample_values_tensor)
            
            # Track max memory length
            max_memory_len = max(max_memory_len, sample_keys_tensor.shape[1])
        
        # Pad all samples to same memory length
        padded_keys = []
        padded_values = []
        
        for sample_keys, sample_values in zip(all_sample_keys, all_sample_values):
            N = sample_keys.shape[1]
            if N < max_memory_len:
                # Pad with zeros
                pad_len = max_memory_len - N
                pad_keys = torch.zeros(1, pad_len, self.memory_dim, device=self.device)
                pad_values = torch.zeros(1, pad_len, self.memory_dim, device=self.device)
                
                sample_keys = torch.cat([sample_keys, pad_keys], dim=1)
                sample_values = torch.cat([sample_values, pad_values], dim=1)
            
            padded_keys.append(sample_keys)
            padded_values.append(sample_values)
        
        # Stack into batch
        memory_keys = torch.cat(padded_keys, dim=0)  # (B, N_max, D_memory)
        memory_values = torch.cat(padded_values, dim=0)  # (B, N_max, D_memory)
        
        return memory_keys, memory_values


class ActionMemoryInferenceManager:
    """
    Manager for continuous memory during inference.
    Uses a fixed-capacity memory bank with eviction.
    """
    
    def __init__(
        self,
        action_dim: int,
        memory_dim: int = 512,
        trajectory_window: int = 10,
        max_memory_capacity: int = 100,
        encoder_type: str = "mlp",
        eviction_strategy: str = "fifo",
        device: str = "cuda",
    ):
        self.action_dim = action_dim
        self.memory_dim = memory_dim
        self.trajectory_window = trajectory_window
        self.device = device
        
        # Initialize trajectory encoder
        self.trajectory_encoder = TrajectoryEncoder(
            action_dim=action_dim,
            hidden_dim=memory_dim,
            output_dim=memory_dim,
            window_size=trajectory_window,
            encoder_type=encoder_type,
        ).to(device)
        self.trajectory_encoder.eval()  # Inference mode
        
        # Initialize memory bank
        self.memory_bank = MemoryBank(
            max_capacity=max_memory_capacity,
            eviction_strategy=eviction_strategy,
        )
        
        # Buffer for recent actions
        self.action_buffer = []
    
    def add_action(self, action: torch.Tensor):
        """
        Add a new action to the buffer and update memory if window is full.
        
        Args:
            action: (D_action,) - single action
        """
        self.action_buffer.append(action)
        
        # Check if we have enough actions to form a trajectory
        if len(self.action_buffer) >= self.trajectory_window:
            # Take last window_size actions
            recent_actions = torch.stack(self.action_buffer[-self.trajectory_window:])  # (window_size, D_action)
            
            # Encode into trajectory latent
            with torch.no_grad():
                traj_latent = self.trajectory_encoder(recent_actions.unsqueeze(0))  # (1, 1, D_memory)
            
            # Add to memory bank
            self.memory_bank.add(traj_latent)
    
    def get_memory(self) -> tuple:
        """
        Get current memory for model input.
        
        Returns:
            memory_keys: (1, N, D_memory) or None
            memory_values: (1, N, D_memory) or None
        """
        return self.memory_bank.get_memory()
    
    def clear(self):
        """Clear memory and action buffer."""
        self.memory_bank.clear()
        self.action_buffer = []


# Example usage in training
def example_training_step(
    model,
    batch_data: Dict,
    memory_wrapper: ActionMemoryTrainingWrapper,
):
    """
    Example of one training step with memory.
    """
    # Extract data
    backbone_output = batch_data['backbone_output']  # BatchFeature
    action_input = batch_data['action_input']  # BatchFeature
    
    # Get full action trajectories (for memory building)
    batch_actions = batch_data['full_actions']  # (B, T_total, D_action)
    
    # Build per-sample memory
    memory_keys, memory_values = memory_wrapper.build_per_sample_memory(
        batch_actions=batch_actions,
    )
    
    # Forward pass with memory
    output = model(
        backbone_output=backbone_output,
        action_input=action_input,
        memory_keys=memory_keys,
        memory_values=memory_values,
    )
    
    return output


# Example usage in inference
def example_inference_rollout(
    model,
    env,
    inference_manager: ActionMemoryInferenceManager,
    max_steps: int = 1000,
):
    """
    Example of inference rollout with continuous memory.
    """
    inference_manager.clear()
    
    obs = env.reset()
    
    for step in range(max_steps):
        # Get backbone output (vision + text processing)
        backbone_output = process_observation(obs)  # Your implementation
        
        # Get current memory
        memory_keys, memory_values = inference_manager.get_memory()
        
        # Get action from model
        with torch.no_grad():
            action_output = model.get_action(
                backbone_output=backbone_output,
                action_input=create_action_input(obs),  # Your implementation
                memory_keys=memory_keys,
                memory_values=memory_values,
            )
        
        action = action_output['action_pred'][0]  # (D_action,)
        
        # Execute action
        obs, reward, done, info = env.step(action.cpu().numpy())
        
        # Add action to memory
        inference_manager.add_action(action)
        
        if done:
            break
    
    return True

