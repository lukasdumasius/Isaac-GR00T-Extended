import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import List, Optional, Tuple

from gr00t.model.action_head.embodiment_encoders import (
    CategorySpecificMLP,
    MultiEmbodimentActionEncoder,
)


class DualStreamChunkCompressor(nn.Module):
    """
    Independent (non-shared) dual-stream encoder for memory:
    - Action encoder (with sinusoidal time, but we force t=0 for clean data)
    - State encoder (category-specific MLP)
    Then fuse (concat + proj) and compress 16-step chunk -> single latent.
    """

    def __init__(
        self,
        action_dim: int,
        state_dim: Optional[int],
        hidden_size: int,
        num_embodiments: int,
        max_seq_len: int = 64,
        nhead: int = 8,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.action_encoder = MultiEmbodimentActionEncoder(
            action_dim=action_dim,
            hidden_size=hidden_size,
            num_embodiments=num_embodiments,
        )
        self.state_encoder = (
            CategorySpecificMLP(
                num_categories=num_embodiments,
                input_dim=state_dim if state_dim is not None else 1,
                hidden_dim=hidden_size,
                output_dim=hidden_size,
            )
            if state_dim is not None and state_dim > 0
            else None
        )

        # Fusion: concat action/state -> project back to hidden_size
        self.fusion_proj = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.SiLU(),
        )

        # Compression: prepend learnable query, transformer encoder, take query output
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_size) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=nhead,
            dim_feedforward=4 * hidden_size,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.compressor = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pos_emb = nn.Parameter(torch.randn(1, max_seq_len + 1, hidden_size) * 0.02)

        self._init_weights()

    def _init_weights(self):
        for p in self.compressor.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self,
        actions: torch.Tensor,
        states: Optional[torch.Tensor],
        cat_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            actions: (B, T, action_dim)
            states:  (B, T, state_dim) or (B, state_dim) or None
            cat_ids: (B,) long or None (default to zeros)
        Returns:
            traj_latent: (B, 1, hidden_size)
            fused_seq:   (B, T, hidden_size) fused token sequence
        """
        B, T, _ = actions.shape
        device = actions.device
        
        # Ensure dtype matches model parameters (e.g., bfloat16 for mixed precision)
        # Get target dtype from action_encoder parameters
        target_dtype = next(self.action_encoder.parameters()).dtype
        if actions.dtype != target_dtype:
            actions = actions.to(dtype=target_dtype)
        if states is not None and states.dtype != target_dtype:
            states = states.to(dtype=target_dtype)

        if cat_ids is None:
            cat_ids = torch.zeros(B, dtype=torch.long, device=device)

        # Force clean timestep for memory encoding
        timesteps = torch.zeros(B, device=device, dtype=torch.long)
        action_feat = self.action_encoder(actions, timesteps, cat_ids)  # (B, T, H)

        if states is None or self.state_encoder is None:
            state_feat = torch.zeros_like(action_feat)
        else:
            # Normalize state shape to (B, T, D_state)
            if states.dim() == 2:
                states = states.unsqueeze(1)  # (B,1,D)
            if states.shape[1] != T:
                # Broadcast or repeat first frame to match T
                if states.shape[1] == 1:
                    states = states.expand(-1, T, -1)
                else:
                    # Truncate or pad with last frame
                    if states.shape[1] > T:
                        states = states[:, :T, :]
                    else:
                        pad_len = T - states.shape[1]
                        pad = states[:, -1:, :].expand(-1, pad_len, -1)
                        states = torch.cat([states, pad], dim=1)
            state_feat = self.state_encoder(states, cat_ids)  # (B, T, H)

        fused = torch.cat([action_feat, state_feat], dim=-1)  # (B, T, 2H)
        fused = self.fusion_proj(fused)  # (B, T, H)

        cls = self.cls_token.expand(B, -1, -1)
        seq = torch.cat([cls, fused], dim=1)  # (B, 1+T, H)

        # Positional embedding slice
        max_L = self.pos_emb.shape[1]
        if seq.shape[1] <= max_L:
            seq = seq + self.pos_emb[:, : seq.shape[1], :]
        else:
            seq = seq + self.pos_emb[:, :max_L, :]

        out = self.compressor(seq)
        traj_latent = out[:, 0:1, :]
        return traj_latent, fused

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
    traj_latent: torch.Tensor       # (1, D_hidden) - stored latent (no batch dim)
    actions: Optional[torch.Tensor] = None # Kept for debugging/visualization only

class ActionMemory(nn.Module):
    def __init__(
        self, 
        action_dim: int,
        action_hidden_size: int,
        state_dim: Optional[int] = None,
        num_embodiments: int = 1,
        max_memory_size: int = 32,
        readout_nhead: int = 4,
    ):
        super().__init__()
        
        # 1. Trajectory Encoder (Dual-stream, independent copy)
        self.encoder = DualStreamChunkCompressor(
            action_dim=action_dim,
            state_dim=state_dim,
            hidden_size=action_hidden_size,
            num_embodiments=num_embodiments,
        )

        # 2. Memory position embedding (fixed relative window, learnable params)
        # Index 0 (oldest) ... max_memory_size-1 (newest)
        self.max_memory_size = max_memory_size
        self.mem_pos_emb = nn.Parameter(torch.randn(1, max_memory_size, action_hidden_size) * 0.02)

        # 3. Memory Readout (Attention + MLP + Gating)
        # Query: current DiT/action latent (D_hidden)
        # Key:   traj_latent + pos_emb + text_emb (no vision)
        # Value: traj_latent + pos_emb (pure motion)
        self.readout = MemoryReadoutBlock(
            query_dim=action_hidden_size,
            memory_key_dim=action_hidden_size,
            memory_val_dim=action_hidden_size,
            hidden_dim=action_hidden_size * 4,
            nhead=readout_nhead
        )
        
        # 4. Memory Bank
        self.memory_bank: List[MemoryEntry] = []
        self.next_id = 0

    def forward(
        self,
        actions: torch.Tensor,
        timesteps: torch.Tensor,
        *,
        states: Optional[torch.Tensor] = None,
        cat_ids: Optional[torch.Tensor] = None,
        text_emb: Optional[torch.Tensor] = None,
        retrieve: bool = False,
    ):
        """
        Args:
            actions: (B, T, action_dim)
            timesteps: (B,)  -- ignored; we force clean timestep inside encoder
            states: optional state tensor aligned with actions (B, T, state_dim)
            cat_ids: optional embodiment ids (B,)
            text_emb: optional text embedding for key construction.
                Accepts shape (B, D_hidden) or (B, 1, D_hidden). If None, treated as zeros.
            retrieve: If True, perform retrieval using the current trajectory latent as query.
        """
        # 1. Encode (dual-stream, clean timestep)
        traj_latent, fused_tokens = self.encoder(actions, states, cat_ids)

        # 2. Retrieval (Optional)
        retrieved_memory = None
        if retrieve and len(self.memory_bank) > 0:
            retrieved_memory = self.perform_retrieval(query_latent=traj_latent, text_emb=text_emb)
        
        return {
            "fused_tokens": fused_tokens,
            "traj_latent": traj_latent,
            "retrieved_memory": retrieved_memory
        }

    def _build_keys_values(
        self, memory_latents: torch.Tensor, text_emb: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Construct keys/values from memory_latents with pos + text.
        memory_latents: (B, N, D)
        returns: keys, vals both (B, N, D)
        """
        B, N, D = memory_latents.shape

        # Relative positional embedding: take last N positions so "newest" aligns到末尾
        pos = self.mem_pos_emb[:, -N:, :].expand(B, -1, -1)  # (B, N, D)

        # Text embedding -> (B, 1, D) then expand to (B, N, D)
        if text_emb is None:
            text = 0.0
        else:
            if text_emb.dim() == 2:
                text = text_emb.unsqueeze(1)  # (B, 1, D)
            else:
                text = text_emb  # (B, 1, D)
            text = text.expand(B, N, -1)

        keys = memory_latents + pos + text
        vals = memory_latents + pos
        return keys, vals

    def build_from_traj_latent(
        self, traj_latent: torch.Tensor, text_emb: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Build K/V directly from the current trajectory latent (no bank).
        traj_latent: (B, 1, D)
        returns: (keys, values) each (B, 1, D)
        """
        memory_latents = traj_latent  # already (B, 1, D)
        keys, vals = self._build_keys_values(memory_latents, text_emb)
        return keys, vals

    def perform_retrieval(self, query_latent: torch.Tensor, text_emb: Optional[torch.Tensor] = None):
        """
        Retrieves relevant memories using the query_latent.
        """
        # Stack memory bank into tensor: (N, D_hidden)
        latents = torch.stack([e.traj_latent for e in self.memory_bank]).to(query_latent.device)  # (N, D)

        # Add Batch dim -> (B, N, D)
        B = query_latent.shape[0]

        memory_latents = latents.unsqueeze(0).expand(B, -1, -1)  # (B, N, D)

        keys, vals = self._build_keys_values(memory_latents, text_emb)
        
        # Run Readout Block
        fused_mem = self.readout(
            current_query=query_latent,
            memory_keys=keys,
            memory_values=vals
        )
        return fused_mem
        
    @torch.no_grad()
    def add_to_memory(self, result_dict, original_actions=None):
        batch_size = result_dict["traj_latent"].shape[0]
        traj = result_dict["traj_latent"].detach().cpu()  # (B, 1, D)
        actions = original_actions.detach().cpu() if original_actions is not None else None
        
        for i in range(batch_size):
            entry = MemoryEntry(
                trajectory_id=self.next_id,
                traj_latent=traj[i].squeeze(0),         # (D_hidden,)
                actions=actions[i] if actions is not None else None
            )
            self.memory_bank.append(entry)
            self.next_id += 1
            
        if len(self.memory_bank) > self.max_memory_size:
            self.memory_bank = self.memory_bank[-self.max_memory_size:]
