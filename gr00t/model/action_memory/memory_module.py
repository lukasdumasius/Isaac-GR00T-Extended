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
            x = x + self.pos_emb[:, :self.pos_emb.shape[1], :]
        
        # 4. Pass through Transformer Encoder
        x_processed = self.compressor(x)
        
        # 5. Extract the [CLS] token output as the trajectory latent
        # (B, 1, H)
        traj_latent = x_processed[:, 0:1, :]
            
        return traj_latent, raw_latents

class MemoryReadoutBlock(nn.Module):
    """
    A full Transformer Decoder Block (Cross-Attn + MLP) to robustly retrieve
    and process memory information. Includes Gating for safe injection.
    """
    def __init__(self, query_dim, memory_key_dim, memory_val_dim, hidden_dim, nhead=4, dropout=0.1):
        super().__init__()
        
        # Projections to align Memory dimensions to Query dimension
        self.key_proj = nn.Linear(memory_key_dim, query_dim)
        self.val_proj = nn.Linear(memory_val_dim, query_dim)
        
        # 1. Cross-Attention
        self.norm1 = nn.LayerNorm(query_dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=query_dim, 
            num_heads=nhead, 
            dropout=dropout, 
            batch_first=True
        )
        
        # 2. Feed-Forward Network (MLP)
        self.norm2 = nn.LayerNorm(query_dim)
        self.ffn = nn.Sequential(
            nn.Linear(query_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, query_dim),
            nn.Dropout(dropout)
        )
        
        # 3. Zero-Initialization Gating (ControlNet Style)
        # Initialize output projection to zero so initial influence is 0
        self.output_gate = nn.Linear(query_dim, query_dim)
        nn.init.zeros_(self.output_gate.weight)
        nn.init.zeros_(self.output_gate.bias)

    def forward(self, current_query, memory_keys, memory_values, key_padding_mask=None):
        """
        Args:
            current_query: (B, L_q, D_q) - e.g. (B, 1, D) current state
            memory_keys:   (B, N, D_k)   - Semantic Latents
            memory_values: (B, N, D_v)   - Compressed/Trajectory Latents
            key_padding_mask: (B, N)     - Mask for padding keys
            
        Returns:
            fused_context: (B, L_q, D_q)
        """
        # Align dimensions
        K = self.key_proj(memory_keys)   # (B, N, D_q)
        V = self.val_proj(memory_values) # (B, N, D_q)
        Q = current_query                # (B, L_q, D_q)
        
        # --- Block 1: Cross Attention ---
        # Pre-Norm
        Q_norm = self.norm1(Q)
        
        # Cross Attn: Q queries K, retrieves V
        attn_out, _ = self.cross_attn(
            query=Q_norm, 
            key=K, 
            value=V, 
            key_padding_mask=key_padding_mask
        )
        
        # Residual 1
        x = Q + attn_out
        
        # --- Block 2: FFN ---
        # Pre-Norm
        x_norm = self.norm2(x)
        ffn_out = self.ffn(x_norm)
        
        # Residual 2
        x = x + ffn_out
        
        # --- Gating ---
        gated_out = self.output_gate(x)
        return gated_out

@dataclass
class MemoryEntry:
    trajectory_id: int
    semantic_latent: torch.Tensor   # (B, 1, D_llm) - Key
    traj_latent: torch.Tensor       # (B, 1, D_hidden) - Value (Compressed)
    value: float                    # Critic score
    actions: Optional[torch.Tensor] = None # Kept for debugging/visualization only

class ActionMemory(nn.Module):
    def __init__(
        self, 
        action_dim: int,
        action_hidden_size: int,
        llm_hidden_size: int,
        critic_hidden_size: int = 256,
        max_memory_size: int = 1000,
        readout_nhead: int = 4
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
        
        # 4. Memory Readout (Attention + MLP + Gating)
        # Query: Current Trajectory Latent (Action Space D_hidden)
        # Key: Semantic Latent (D_llm)
        # Value: Trajectory Latent (D_hidden)
        # We output in D_hidden space to fuse back into policy
        self.readout = MemoryReadoutBlock(
            query_dim=action_hidden_size,
            memory_key_dim=llm_hidden_size,
            memory_val_dim=action_hidden_size,
            hidden_dim=action_hidden_size * 4,
            nhead=readout_nhead
        )
        
        # 5. Memory Bank
        self.memory_bank: List[MemoryEntry] = []
        self.max_memory_size = max_memory_size
        self.next_id = 0

    def forward(self, actions, timesteps, llm_backbone_fn=None, retrieve=False):
        """
        Args:
            retrieve: If True, perform retrieval using the current action latent as query.
        """
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
        
        # 5. Retrieval (Optional)
        retrieved_memory = None
        if retrieve and len(self.memory_bank) > 0:
            retrieved_memory = self.perform_retrieval(query_latent=traj_latent)
        
        return {
            "value": value_pred,
            "semantic_latent": semantic_latent,
            "raw_latents": raw_latents, # Still returned for loss computation if needed, but not stored
            "traj_latent": traj_latent,
            "retrieved_memory": retrieved_memory
        }
        
    def perform_retrieval(self, query_latent):
        """
        Retrieves relevant memories using the query_latent.
        """
        # Stack memory bank into tensors
        # Keys: (B, N, D_llm)
        # Vals: (B, N, D_hidden)
        
        keys = torch.stack([e.semantic_latent.squeeze(0).squeeze(0) for e in self.memory_bank]).to(query_latent.device)
        vals = torch.stack([e.traj_latent.squeeze(0).squeeze(0) for e in self.memory_bank]).to(query_latent.device)
        
        # Add Batch dim (1, N, D) -> Expand to (B, N, D)
        B = query_latent.shape[0]
        keys = keys.unsqueeze(0).expand(B, -1, -1)
        vals = vals.unsqueeze(0).expand(B, -1, -1)
        
        # Run Readout Block
        fused_mem = self.readout(
            current_query=query_latent,
            memory_keys=keys,
            memory_values=vals
        )
        return fused_mem

    def compute_critic_loss(self, predicted_value, target_value):
        """
        Computes MSE loss for the critic.
        """
        if predicted_value.dim() == 3:
            predicted_value = predicted_value.squeeze(1)
        if target_value.dim() == 1:
            target_value = target_value.unsqueeze(1)
            
        return F.mse_loss(predicted_value, target_value)

    def project_to_llm(self, traj_latent):
        return self.projector(traj_latent)
        
    @torch.no_grad()
    def add_to_memory(self, result_dict, original_actions=None):
        batch_size = result_dict["semantic_latent"].shape[0]
        semantic = result_dict["semantic_latent"].detach().cpu()
        traj = result_dict["traj_latent"].detach().cpu() 
        values = result_dict["value"].detach().cpu()
        actions = original_actions.detach().cpu() if original_actions is not None else None
        
        for i in range(batch_size):
            entry = MemoryEntry(
                trajectory_id=self.next_id,
                semantic_latent=semantic[i], # (1, D_llm)
                traj_latent=traj[i],         # (1, D_hidden)
                value=values[i].item(),
                actions=actions[i] if actions is not None else None
            )
            self.memory_bank.append(entry)
            self.next_id += 1
            
        if len(self.memory_bank) > self.max_memory_size:
            self.memory_bank = self.memory_bank[-self.max_memory_size:]
