"""IMPALA-style ResNet actor-critic, as used by the strong PPO baseline of Moon et al. (2023).

Compared with the 2022 networks (networks/actor_network.py, critic_network.py):
  - one shared encoder for the policy and the value (two small heads), instead of two
    separate CNNs;
  - a deeper encoder: 3 stacks of [conv 3x3 -> max-pool /2 -> 2 residual blocks], with
    64, 128, 128 channels (64x64 -> 32 -> 16 -> 8), then dense layers 8192 -> 256 -> 1024;
  - normalisation (GroupNorm with 1 group = LayerNorm over C,H,W for convolutions, LayerNorm for
    dense layers) before each layer, which keeps activations at a stable scale in a deep net;
  - about 4M parameters (vs ~0.6M per 2022 network).
"""
import os

import torch
from torch import nn
from torch.distributions import Categorical


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm0 = nn.GroupNorm(1, channels)
        self.conv0 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(1, channels)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x):
        y = self.conv0(torch.relu(self.norm0(x)))
        y = self.conv1(torch.relu(self.norm1(y)))
        return x + y


class ConvStack(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.norm = nn.GroupNorm(1, in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.pool = nn.MaxPool2d(3, stride=2, padding=1)
        self.res0 = ResidualBlock(out_channels)
        self.res1 = ResidualBlock(out_channels)

    def forward(self, x):
        x = self.pool(self.conv(self.norm(x)))
        return self.res1(self.res0(x))


class ImpalaActorCritic(nn.Module):
    def __init__(self, n_actions, input_dims, lr, channels=(64, 128, 128), chkpt_dir='checkpoints/ppo'):
        super().__init__()
        self.checkpoint_file = os.path.join(chkpt_dir, 'impala_torch_ppo')
        c, h, w = input_dims
        stacks = []
        for out in channels:
            stacks.append(ConvStack(c, out))
            c, h, w = out, (h + 1) // 2, (w + 1) // 2
        self.encoder = nn.Sequential(*stacks)
        flat = c * h * w                                  # 128 * 8 * 8 = 8192 for 64x64 input
        self.dense = nn.Sequential(
            nn.ReLU(), nn.Flatten(),
            nn.LayerNorm(flat), nn.Linear(flat, 256), nn.ReLU(),
            nn.LayerNorm(256), nn.Linear(256, 1024), nn.ReLU(),
            nn.LayerNorm(1024),
        )
        self.policy_head = nn.Linear(1024, n_actions)
        self.value_head = nn.Linear(1024, 1)
        # small initial outputs: near-uniform policy and values near 0 at the start
        nn.init.orthogonal_(self.policy_head.weight, gain=0.01)
        nn.init.zeros_(self.policy_head.bias)
        nn.init.orthogonal_(self.value_head.weight, gain=0.1)
        nn.init.zeros_(self.value_head.bias)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        """Returns (action distribution, value [B])."""
        features = self.dense(self.encoder(state))
        return Categorical(logits=self.policy_head(features)), self.value_head(features).reshape(-1)

    def save_checkpoint(self):
        os.makedirs(os.path.dirname(self.checkpoint_file), exist_ok=True)
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file, map_location=self.device))
