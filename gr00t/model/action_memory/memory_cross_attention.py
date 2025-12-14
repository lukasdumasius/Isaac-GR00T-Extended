"""
Memory Cross-Attention Module for Action Memory Integration
Simplified version: Direct trajectory latent retrieval without LLM processing
"""

import torch
import torch.nn as nn
from typing import Optional, List, Dict
import torch.nn.functional as F


class MemoryCrossAttention(nn.Module):
    """
    Cross-attention module for retrieving relevant trajectory memories.
    
    Query: Action latent from DiT block
    Key: Trajectory latents from memory bank
    Value: Trajectory latents from memory bank
    """
    
    def __init__(
        self,
        query_dim: int,
        memory_dim: int,
        num_heads: int = 8,
        dropout: float = 0.0,
        bias: bool = True,
        gated: bool = False,  # Future: gated cross-attention
    ):
        super().__init__()
        self.query_dim = query_dim
        self.memory_dim = memory_dim
        self.num_heads = num_heads
        self.head_dim = query_dim // num_heads
        self.gated = gated
        
        assert query_dim % num_heads == 0, f"query_dim {query_dim} must be divisible by num_heads {num_heads}"
        
        # Q, K, V projection layers
        self.to_q = nn.Linear(query_dim, query_dim, bias=bias)
        self.to_k = nn.Linear(memory_dim, query_dim, bias=bias)
        self.to_v = nn.Linear(memory_dim, query_dim, bias=bias)
        
        # Output projection
        self.to_out = nn.Sequential(
            nn.Linear(query_dim, query_dim, bias=bias),
            nn.Dropout(dropout)
        )
        
        # Gating mechanism (future work)
        if gated:
            self.gate = nn.Parameter(torch.zeros(1))
        else:
            self.gate = None
            
        # Layer norm for stability
        self.norm = nn.LayerNorm(query_dim)
        
    def forward(
        self,
        query: torch.Tensor,  # (B, T, query_dim) - from DiT block
        memory_keys: Optional[torch.Tensor] = None,  # (B, N, memory_dim) - trajectory latents
        memory_values: Optional[torch.Tensor] = None,  # (B, N, memory_dim) - trajectory latents
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            query: Current action latent from DiT block, shape (B, T, query_dim)
            memory_keys: Trajectory latents from memory bank, shape (B, N, memory_dim)
            memory_values: Trajectory latents from memory bank, shape (B, N, memory_dim)
            attention_mask: Optional mask for attention, shape (B, T, N)
            
        Returns:
            Retrieved memory, shape (B, T, query_dim)
        """
        B, T, _ = query.shape
        
        # If no memory, return zeros
        if memory_keys is None or memory_keys.size(1) == 0:
            return torch.zeros_like(query)
        
        N = memory_keys.size(1)
        
        # Project to Q, K, V
        q = self.to_q(query)  # (B, T, query_dim)
        k = self.to_k(memory_keys)  # (B, N, query_dim)
        v = self.to_v(memory_values)  # (B, N, query_dim)
        
        # Reshape for multi-head attention
        # (B, T, query_dim) -> (B, num_heads, T, head_dim)
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        # (B, num_heads, T, head_dim) @ (B, num_heads, head_dim, N) -> (B, num_heads, T, N)
        attention_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        
        # Apply attention mask if provided
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask
        
        # Softmax
        attention_probs = F.softmax(attention_scores, dim=-1)
        
        # Apply attention to values
        # (B, num_heads, T, N) @ (B, num_heads, N, head_dim) -> (B, num_heads, T, head_dim)
        attended = torch.matmul(attention_probs, v)
        
        # Reshape back
        # (B, num_heads, T, head_dim) -> (B, T, query_dim)
        attended = attended.transpose(1, 2).contiguous().view(B, T, self.query_dim)
        
        # Output projection
        output = self.to_out(attended)
        
        # Apply gating if enabled
        if self.gated and self.gate is not None:
            output = torch.sigmoid(self.gate) * output
        
        return output


class MemoryBank:
    """
    Simple memory bank for storing and retrieving trajectory latents.
    Simplified version: No LLM processing, direct trajectory latent storage.
    """
    
    def __init__(
        self,
        max_capacity: int = 100,
        eviction_strategy: str = "fifo",  # "fifo", "random", "attention"
    ):
        self.max_capacity = max_capacity
        self.eviction_strategy = eviction_strategy
        self.memory_keys = []  # List of trajectory latents
        self.memory_values = []  # Same as keys in simplified version
        
    def add(self, trajectory_latent: torch.Tensor):
        """
        Add a trajectory latent to memory.
        
        Args:
            trajectory_latent: Tensor of shape (1, 1, D_hidden)
        """
        # Check capacity
        if len(self.memory_keys) >= self.max_capacity:
            self._evict()
        
        # Add to memory (keys and values are the same in simplified version)
        self.memory_keys.append(trajectory_latent)
        self.memory_values.append(trajectory_latent)
    
    def _evict(self):
        """Evict one memory entry based on strategy."""
        if self.eviction_strategy == "fifo":
            self.memory_keys.pop(0)
            self.memory_values.pop(0)
        elif self.eviction_strategy == "random":
            import random
            idx = random.randint(0, len(self.memory_keys) - 1)
            self.memory_keys.pop(idx)
            self.memory_values.pop(idx)
        else:
            # Default to FIFO
            self.memory_keys.pop(0)
            self.memory_values.pop(0)
    
    def get_memory(self) -> tuple:
        """
        Get all memory keys and values as tensors.
        
        Returns:
            (keys, values): Both of shape (1, N, D_hidden) or None if empty
        """
        if len(self.memory_keys) == 0:
            return None, None
        
        # Stack all memories along sequence dimension
        keys = torch.cat(self.memory_keys, dim=1)  # (1, N, D_hidden)
        values = torch.cat(self.memory_values, dim=1)  # (1, N, D_hidden)
        
        return keys, values
    
    def clear(self):
        """Clear all memory."""
        self.memory_keys = []
        self.memory_values = []
    
    def __len__(self):
        return len(self.memory_keys)


class TrajectoryEncoder(nn.Module):
    """
    Encode a sequence of actions into a compact trajectory latent.
    
    Uses a simple MLP or 1D conv to compress action sequences.
    """
    
    def __init__(
        self,
        action_dim: int,
        hidden_dim: int,
        output_dim: int,
        window_size: int = 10,
        encoder_type: str = "mlp",  # "mlp" or "conv"
    ):
        super().__init__()
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.window_size = window_size
        self.encoder_type = encoder_type
        
        if encoder_type == "mlp":
            # Flatten window and encode
            self.encoder = nn.Sequential(
                nn.Linear(action_dim * window_size, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, output_dim),
            )
        elif encoder_type == "conv":
            # 1D convolution over time
            self.encoder = nn.Sequential(
                nn.Conv1d(action_dim, hidden_dim, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(1),  # Pool to single timestep
                nn.Flatten(),
                nn.Linear(hidden_dim, output_dim),
            )
        else:
            raise ValueError(f"Unknown encoder_type: {encoder_type}")
    
    def forward(self, actions: torch.Tensor) -> torch.Tensor:
        """
        Encode a sequence of actions into a trajectory latent.
        
        Args:
            actions: Tensor of shape (B, window_size, action_dim)
            
        Returns:
            trajectory_latent: Tensor of shape (B, 1, output_dim)
        """
        B, T, D = actions.shape
        
        # Ensure actions are in the same dtype as encoder weights
        # Get the dtype of the first parameter in encoder
        encoder_dtype = next(self.encoder.parameters()).dtype
        if actions.dtype != encoder_dtype:
            actions = actions.to(encoder_dtype)
        
        if self.encoder_type == "mlp":
            # Flatten time and action dimensions
            actions_flat = actions.view(B, -1)  # (B, window_size * action_dim)
            latent = self.encoder(actions_flat)  # (B, output_dim)
            latent = latent.unsqueeze(1)  # (B, 1, output_dim)
        elif self.encoder_type == "conv":
            # Transpose for conv1d: (B, action_dim, window_size)
            actions_t = actions.transpose(1, 2)  # (B, action_dim, window_size)
            latent = self.encoder(actions_t)  # (B, output_dim)
            latent = latent.unsqueeze(1)  # (B, 1, output_dim)
        
        return latent

