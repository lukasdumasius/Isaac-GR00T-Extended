import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from gr00t.model.action_head.action_encoder import ActionEncoder
from .critic import TrajectoryCritic

class TrajectoryCompressor(nn.Module):
    """
    Wraps the existing ActionEncoder and applies a Transformer-based compression
    to generate a single 'trajectory latent' (concept) from a sequence.
    """
    def __init__(self, action_dim, hidden_size, num_layers=2, nhead=4, max_seq_len=100):
        super().__init__()
        # Reuse the existing ActionEncoder logic
        self.action_encoder = ActionEncoder(action_dim, hidden_size)
        self.hidden_size = hidden_size
        
        # Learnable [CLS] token to aggregate trajectory information
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_size) * 0.02)
        
        # Positional Embedding for [CLS] + Sequence
        # +1 for the [CLS] token
        self.pos_emb = nn.Parameter(torch.randn(1, max_seq_len + 1, hidden_size) * 0.02)
        
        # Transformer Encoder to process the sequence and aggregate into [CLS]
        # batch_first=True is important for (B, T, D) inputs
        # norm_first=True (Pre-LN) is generally more stable for training
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size, 
            nhead=nhead, 
            dim_feedforward=4*hidden_size,
            dropout=0.1,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.compressor = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self._init_weights()

    def _init_weights(self):
        """Initialize weights for stability."""
        for p in self.compressor.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, actions, timesteps):
        """
        Args:
            actions: (B, T, action_dim)
            timesteps: (B,) 
        Returns:
            traj_latent: (B, 1, hidden_size) - The processed [CLS] token
            raw_latents: (B, T, hidden_size) - The per-step latents from base encoder
        """
        batch_size = actions.shape[0]
        T = actions.shape[1]
        
        # 1. Get raw per-step latents from the base encoder
        # Output: (B, T, hidden_size)
        raw_latents = self.action_encoder(actions, timesteps)
        
        # 2. Append [CLS] token to the BEGINNING of the sequence
        # Expand CLS to batch size: (B, 1, H)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        
        # Concat: (B, 1 + T, H)
        # [CLS, t_0, t_1, ..., t_T]
        x = torch.cat((cls_tokens, raw_latents), dim=1)
        
        # 3. Add Positional Embedding
        # We slice pos_emb to match current sequence length (1 + T)
        # Assuming batch broadcasting works: (1, L, D) + (B, L, D)
        if T + 1 <= self.pos_emb.shape[1]:
            x = x + self.pos_emb[:, :T+1, :]
        else:
            # Fallback for overly long sequences (just use max available pos embs)
            # Alternatively, we could interpolate or warning.
            x = x + self.pos_emb[:, :self.pos_emb.shape[1], :]
        
        # 4. Pass through Transformer Encoder
        x_processed = self.compressor(x)
        
        # 5. Extract the [CLS] token output as the trajectory latent
        # (B, 1, H)
        traj_latent = x_processed[:, 0:1, :]
            
        return traj_latent, raw_latents

@dataclass
class MemoryEntry:
    trajectory_id: int
    raw_latent: torch.Tensor        # (B, T, D_action)
    semantic_latent: torch.Tensor   # (B, 1, D_llm)
    value: float                    # Critic score
    actions: Optional[torch.Tensor] = None 

class ActionMemory(nn.Module):
    def __init__(
        self, 
        action_dim: int,
        action_hidden_size: int,
        llm_hidden_size: int,
        critic_hidden_size: int = 256,
        aggregation: str = "attention", # Kept for compatibility, but effectively unused with new Compressor
        max_memory_size: int = 1000
    ):
        super().__init__()
        
        # 1. Trajectory Encoder (Transformer-based)
        self.encoder = TrajectoryCompressor(
            action_dim=action_dim, 
            hidden_size=action_hidden_size
        )
        
        # 2. Projector (Action Latent -> LLM Space)
        self.projector = nn.Linear(action_hidden_size, llm_hidden_size)
        
        # 3. Critic (LLM Space -> Value)
        self.critic = TrajectoryCritic(
            input_dim=llm_hidden_size,
            hidden_dim=critic_hidden_size
        )
        
        # 4. Memory Bank
        self.memory_bank: List[MemoryEntry] = []
        self.max_memory_size = max_memory_size
        self.next_id = 0

    def forward(self, actions, timesteps, llm_backbone_fn=None):
        # 1. Encode
        traj_latent, raw_latents = self.encoder(actions, timesteps)
        
        # 2. Project
        llm_input = self.project_to_llm(traj_latent)
        
        # 3. LLM Processing (Injection)
        if llm_backbone_fn is not None:
            semantic_latent = llm_backbone_fn(llm_input)
        else:
            semantic_latent = llm_input
            
        # 4. Critic Evaluation
        value_pred = self.critic(semantic_latent)
        
        return {
            "value": value_pred,
            "semantic_latent": semantic_latent,
            "raw_latents": raw_latents,
            "traj_latent": traj_latent
        }
        
    def compute_critic_loss(self, predicted_value, target_value):
        """
        Computes MSE loss for the critic.
        
        Args:
            predicted_value: (B, 1, 1) or (B, 1) output from self.critic
            target_value: (B, 1) or (B,) ground truth values (e.g., success=1, fail=0)
            
        Returns:
            loss: scalar tensor
        """
        # Ensure shapes align
        if predicted_value.dim() == 3:
            predicted_value = predicted_value.squeeze(1) # (B, 1)
        if target_value.dim() == 1:
            target_value = target_value.unsqueeze(1) # (B, 1)
            
        return F.mse_loss(predicted_value, target_value)

    def project_to_llm(self, traj_latent):
        return self.projector(traj_latent)
        
    @torch.no_grad()
    def add_to_memory(self, result_dict, original_actions=None):
        batch_size = result_dict["semantic_latent"].shape[0]
        semantic = result_dict["semantic_latent"].detach().cpu()
        raw = result_dict["raw_latents"].detach().cpu()
        values = result_dict["value"].detach().cpu()
        actions = original_actions.detach().cpu() if original_actions is not None else None
        
        for i in range(batch_size):
            entry = MemoryEntry(
                trajectory_id=self.next_id,
                raw_latent=raw[i],
                semantic_latent=semantic[i],
                value=values[i].item(),
                actions=actions[i] if actions is not None else None
            )
            self.memory_bank.append(entry)
            self.next_id += 1
            
        if len(self.memory_bank) > self.max_memory_size:
            self.memory_bank = self.memory_bank[-self.max_memory_size:]
