import torch


class RandomAgent:
    def __init__(self, action_num) -> None:
        self.action_num = action_num
        # a uniformly random policy
        self.policy = torch.distributions.Categorical(
            torch.ones(action_num) / action_num
        )

    def act(self, observation, train=None):
        return self.policy.sample().item()