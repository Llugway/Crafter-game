import numpy as np
import torch
import torch.nn.functional as F
from networks.actor_network import ActorNetwork
from networks.critic_network import CriticNetwork
from memories.ppo_memory import PPOMemory


class PPO_Agent:
    def __init__(self, n_actions, input_dims, gamma=0.99, lr=0.0003, gae_lambda=0.95,
                 policy_clip=0.2, batch_size=64, n_epochs=10, n_rollouts=2048, max_grad_norm=0.5,
                 chkpt_dir='checkpoints/ppo'):
        self.gamma = gamma
        self.policy_clip = policy_clip
        self.n_epochs = n_epochs
        self.gae_lambda = gae_lambda
        self.max_grad_norm = max_grad_norm

        # Actor/Critic policies
        self.actor = ActorNetwork(n_actions, input_dims, lr, chkpt_dir=chkpt_dir)
        self.critic = CriticNetwork(input_dims, lr, chkpt_dir=chkpt_dir)

        # Memory for each rollout
        self.memory = PPOMemory(batch_size, input_dims, n_rollouts)

        # vars for logs
        self.avg_critic_loss = 0
        self.avg_actor_loss = 0
        self.total_loss = 0

    def remember(self, state, action, probs, vals, reward, done):
        self.memory.store_memory(state, action, probs, vals, reward, done)

    def save_models(self):
        print('... saving models ...')
        self.actor.save_checkpoint()
        self.critic.save_checkpoint()

    def load_models(self):
        print('... loading models ...')
        self.actor.load_checkpoint()
        self.critic.load_checkpoint()

    def act(self, observation):
        state = observation.to(self.actor.device)

        dist = self.actor(state)
        value = self.critic(state)
        action = dist.sample()

        probs = torch.squeeze(dist.log_prob(action)).item()
        action = torch.squeeze(action).item()
        value = torch.squeeze(value).item()

        return action, probs, value

    def learn(self):

        actor_losses = []
        critic_losses = []

        for _ in range(self.n_epochs):
            state_arr, action_arr, old_prob_arr, vals_arr, \
            reward_arr, dones_arr, batches = \
                self.memory.generate_batches()

            values = vals_arr
            advantage = np.zeros(len(reward_arr), dtype=np.float32)

            for t in range(len(reward_arr) - 1):
                discount = 1
                a_t = 0
                for k in range(t, len(reward_arr) - 1):
                    a_t += discount * (reward_arr[k] + self.gamma * values[k + 1] * \
                                       (1 - int(dones_arr[k])) - values[k])
                    discount *= self.gamma * self.gae_lambda
                advantage[t] = a_t
            advantage = torch.tensor(advantage).to(self.actor.device)

            values = torch.tensor(values).to(self.actor.device)
            for batch in batches:
                inter = state_arr[batch]
                states = torch.tensor(inter).to(self.actor.device)
                old_probs = torch.tensor(old_prob_arr[batch]).to(self.actor.device)
                actions = torch.tensor(action_arr[batch]).to(self.actor.device)

                dist = self.actor(states)
                critic_value = self.critic(states)
                critic_value = torch.squeeze(critic_value)

                new_probs = dist.log_prob(actions)
                prob_ratio = new_probs.exp() / old_probs.exp()

                weighted_probs = advantage[batch] * prob_ratio
                weighted_clipped_probs = torch.clamp(prob_ratio, 1 - self.policy_clip,
                                                     1 + self.policy_clip) * advantage[batch]
                actor_loss = -torch.min(weighted_probs, weighted_clipped_probs).mean()
                actor_losses.append(actor_loss.item())

                returns = advantage[batch] + values[batch]

                # Critic loss
                critic_loss = (returns - critic_value) ** 2
                critic_loss = critic_loss.mean()
                critic_losses.append(critic_loss.item())

                total_loss = actor_loss + 0.5 * critic_loss  # Entropy ?

                self.actor.optimizer.zero_grad()
                self.critic.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)  # MAX_GRAD_NORM
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.actor.optimizer.step()
                self.critic.optimizer.step()

        self.memory.clear_memory()

        # Losses logs
        self.avg_critic_loss = np.mean(critic_losses)
        self.avg_actor_loss = np.mean(actor_losses)
        self.total_loss = total_loss.item()

