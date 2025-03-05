import torch


class REINFORCE:
    def __init__(self, gamma = 0.99):
        super().__init__()
        self.gamma = gamma
        self.rewards = []
        self.action_log_probs = []

    def update(self, reward, action_log_prob):
        self.rewards.append(reward)
        self.action_log_probs.append(action_log_prob)

    def reset(self):
        self.rewards.clear()
        self.action_log_probs.clear()

    def get_loss(self):
        log_probs = torch.stack(self.action_log_probs)
        rewards = torch.tensor(self.rewards).to(log_probs)
        
        discounts = self.gamma ** torch.arange(len(self.rewards)).to(rewards)
        cum_dis_rewards = torch.flip(torch.cumsum(rewards * discounts, -1), (-1,))
        normalized_rewards = (cum_dis_rewards - cum_dis_rewards.mean()) / (cum_dis_rewards.std() + 1e-6)
        return -torch.sum(normalized_rewards * log_probs), rewards.sum()
