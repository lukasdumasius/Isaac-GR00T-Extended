import torch
import torch.nn as nn

class TrajectoryCritic(nn.Module):
    """
    A lightweight critic head that evaluates a trajectory latent 
    (after it has been processed by the LLM backbone).
    """
    def __init__(self, input_dim, hidden_dim=256, output_dim=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),  # Swish
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, output_dim)
        )

    def forward(self, llm_traj_latent):
        """
        Args:
            llm_traj_latent: (B, 1, input_dim) or (B, input_dim)
        Returns:
            value: (B, 1, output_dim)
        """
        return self.net(llm_traj_latent)

