import torch
import torch.nn as nn


class CategoricalPolicy(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, action_dim)
        )
        self.distribution = torch.distributions.Categorical

    def forward(self, x):
        x = torch.tensor(x, dtype=torch.float32, device=next(self.parameters()).device)
        x = self.net(x)
        log_probs = nn.functional.log_softmax(x, dim=-1)
        return log_probs

    def sample(self, log_probs):
        action_dist = self.distribution(logits=log_probs)
        action = action_dist.sample()
        log_prob = action_dist.log_prob(action)
        return action.item(), log_prob




