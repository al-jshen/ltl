import numpy as np
import torch


class REINFORCE:
    def __init__(self, gamma = 0.99):
        super().__init__()
        self.gamma = gamma
        self.rewards = []
        self.actions = []
        self.action_log_probs = []

    def update(self, reward, action, action_log_prob):
        self.rewards.append(reward)
        self.actions.append(action)
        self.action_log_probs.append(action_log_prob)

    def reset(self):
        self.rewards.clear()
        self.actions.clear()
        self.action_log_probs.clear()

    def get_loss(self):
        log_probs = torch.stack(self.action_log_probs).T
        log_probs = torch.nan_to_num(log_probs, -1000.)
        rewards = torch.from_numpy(np.stack(self.rewards)).to(log_probs).T
        
        discounts = self.gamma ** torch.arange(rewards.shape[1]).to(rewards)
        cum_dis_rewards = torch.flip(torch.cumsum(rewards * discounts, -1), (-1,))
        # normalized_rewards = (cum_dis_rewards - cum_dis_rewards.mean()) / (cum_dis_rewards.std() + 1e-6) # next line seems better
        if cum_dis_rewards.shape[0] > 1:
            normalized_rewards = (cum_dis_rewards - cum_dis_rewards.mean(axis=0, keepdim=True)) / (cum_dis_rewards.std(axis=0, keepdim=True) + 1e-6)
        else:
            normalized_rewards = cum_dis_rewards
        return -torch.sum(normalized_rewards * log_probs, axis=-1).mean(axis=0), rewards.sum(axis=-1)
