import torch
import torch.nn as nn

from .flow import RealNVP


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
        return action.detach().cpu().numpy(), log_prob


class FlowPolicy(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim, cond_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, cond_dim),
        )
        self.flow = RealNVP(data_dim=action_dim, context_dim=cond_dim, activation="GELU", hidden_dims=[hidden_dim, hidden_dim])

    def forward(self, x):
        x = torch.tensor(x, dtype=torch.float32, device=next(self.parameters()).device)
        conditioner = self.net(x)
        return conditioner

    def sample(self, conditioner):
        action = self.flow.sample(1, context=conditioner).squeeze(0)
        log_prob = self.flow.log_prob(action, context=conditioner)
        return action.detach().cpu().numpy(), log_prob


