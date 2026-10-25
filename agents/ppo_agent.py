import numpy as np
import torch
import torch.nn.functional as F
from networks.actor_network import ActorNetwork
from networks.critic_network import CriticNetwork
from memories.ppo_memory import PPOMemory


class PPO_Agent:
    def __init__(self, n_actions, input_dims, gamma=0.99, lr=0.0003, gae_lambda=0.95,
                 policy_clip=0.2, batch_size=64, n_epochs=10, n_rollouts=2048, max_grad_norm=0.5,
                 chkpt_dir='checkpoints/ppo', ent_coef=0.0):
        self.gamma = gamma
        # Weight of the entropy bonus in the loss. 0.0 = state 0 (2022 code, no bonus).
        self.ent_coef = ent_coef
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
        self.avg_entropy = 0

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

    def _advantages(self, reward_arr, values, dones_arr):
        """a_t = delta_t + gamma*lambda*a_{t+1}, delta_t = r_t + gamma*V(s_{t+1})*(1-done_t) - V(s_t)."""
        r = np.asarray(reward_arr, dtype=np.float64)
        v = np.asarray(values, dtype=np.float64)
        d = np.asarray(dones_arr, dtype=np.float64)
        adv = np.zeros(len(r), dtype=np.float64)
        a = 0.0
        for t in range(len(r) - 2, -1, -1):
            delta = r[t] + self.gamma * v[t + 1] * (1 - d[t]) - v[t]
            a = delta + self.gamma * self.gae_lambda * a
            adv[t] = a
        return adv.astype(np.float32)

    def learn(self):

        actor_losses = []
        critic_losses = []
        entropies = []

        # Advantages (GAE). They depend only on the rollout, not on the epoch, so they are
        # computed once. The original code recomputed them at every epoch with a double loop
        # (O(n^2), ~2M iterations, ~20 s per rollout). This backward recursion gives exactly the
        # same numbers (checked bit-for-bit), including the original behaviour that the
        # discount chain is *not* cut at episode boundaries and the last step gets 0.
        _, _, _, vals_arr, reward_arr, dones_arr, _ = self.memory.generate_batches()
        advantage = self._advantages(reward_arr, vals_arr, dones_arr)

        for _ in range(self.n_epochs):
            state_arr, action_arr, old_prob_arr, vals_arr, \
            reward_arr, dones_arr, batches = \
                self.memory.generate_batches()

            values = vals_arr
            advantage = torch.as_tensor(advantage).to(self.actor.device)

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

                # Entropy of the policy on this mini-batch: high = actions spread out (exploring),
                # low = almost always the same action. Logged in every run, used in the loss
                # only when ent_coef > 0 (subtracted, so minimising the loss increases entropy).
                entropy = dist.entropy().mean()
                entropies.append(entropy.item())

                total_loss = actor_loss + 0.5 * critic_loss - self.ent_coef * entropy

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
        self.avg_entropy = np.mean(entropies)

