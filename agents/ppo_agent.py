import numpy as np
import torch
import torch.nn.functional as F
from networks.actor_network import ActorNetwork
from networks.critic_network import CriticNetwork
from networks.impala import ImpalaActorCritic


class ValueNormalizer:
    """Running mean / std of the value targets (returns), updated with an exponential moving
    average (decay 0.99) once per rollout, with the usual start-up bias correction.

    The critic then predicts *normalised* values: its target always has roughly zero mean and
    unit variance, whatever the scale of the returns (which grows as the agent unlocks more
    achievements). Values are de-normalised before they are used for the advantages."""

    def __init__(self, decay=0.99):
        self.decay, self.count = decay, 0
        self.mean, self.sq = 0.0, 0.0

    def update(self, returns):
        self.count += 1
        self.mean = self.decay * self.mean + (1 - self.decay) * float(returns.mean())
        self.sq = self.decay * self.sq + (1 - self.decay) * float((returns ** 2).mean())

    def stats(self):
        if self.count == 0:
            return 0.0, 1.0
        corr = 1 - self.decay ** self.count
        mean, sq = self.mean / corr, self.sq / corr
        return mean, max(sq - mean ** 2, 1e-4) ** 0.5
from memories.ppo_memory import PPOMemory


class PPO_Agent:
    def __init__(self, n_actions, input_dims, gamma=0.99, lr=0.0003, gae_lambda=0.95,
                 policy_clip=0.2, batch_size=64, n_epochs=10, n_rollouts=2048, max_grad_norm=0.5,
                 chkpt_dir='checkpoints/ppo', ent_coef=0.0,
                 norm_adv=False, gae_cut_at_done=False, bootstrap=False,
                 arch='nature', value_norm=False):
        self.gamma = gamma
        # arch: 'nature' = the 2022 networks (separate actor and critic CNNs),
        #       'impala' = shared IMPALA ResNet encoder with two heads (networks/impala.py)
        # value_norm: the critic learns normalised value targets (ValueNormalizer)
        self.arch = arch
        self.value_norm = ValueNormalizer() if value_norm else None
        # Correctness fixes, all off by default (= 2022 behaviour):
        #   norm_adv         normalise advantages per mini-batch in the policy loss
        #   gae_cut_at_done  stop the GAE sum at the end of an episode
        #   bootstrap        use V(next state) for the last step of a rollout instead of 0
        self.norm_adv = norm_adv
        self.gae_cut_at_done = gae_cut_at_done
        self.bootstrap = bootstrap
        # Weight of the entropy bonus in the loss. 0.0 = state 0 (2022 code, no bonus).
        self.ent_coef = ent_coef
        self.policy_clip = policy_clip
        self.n_epochs = n_epochs
        self.gae_lambda = gae_lambda
        self.max_grad_norm = max_grad_norm

        # Actor/Critic policies
        if arch == 'impala':
            self.model = ImpalaActorCritic(n_actions, input_dims, lr, chkpt_dir=chkpt_dir)
            self.networks = [self.model]
        else:
            self.model = None
            self.actor = ActorNetwork(n_actions, input_dims, lr, chkpt_dir=chkpt_dir)
            self.critic = CriticNetwork(input_dims, lr, chkpt_dir=chkpt_dir)
            self.networks = [self.actor, self.critic]
        self.device = self.networks[0].device
        n_params = sum(p.numel() for net in self.networks for p in net.parameters())
        print(f"Network: {arch}, {n_params / 1e6:.2f}M parameters")

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
        for net in self.networks:
            net.save_checkpoint()

    def load_models(self):
        print('... loading models ...')
        for net in self.networks:
            net.load_checkpoint()

    def _policy_value(self, states):
        """Action distribution and value [B] (normalised if value_norm is on)."""
        if self.model is not None:
            return self.model(states)
        return self.actor(states), self.critic(states).reshape(-1)

    def _denorm(self, values):
        if self.value_norm is None:
            return values
        mean, std = self.value_norm.stats()
        return values * std + mean

    def act(self, observation):
        # no_grad: acting does not need gradients (the 2022 code built a graph at every step).
        # One stack + tolist = one GPU->CPU transfer instead of three .item() calls.
        with torch.no_grad():
            state = observation.to(self.device)
            dist, value = self._policy_value(state)
            value = self._denorm(value)
            action = dist.sample()
            out = torch.stack([action.reshape(()).float(),
                               dist.log_prob(action).reshape(()),
                               value.reshape(())]).tolist()
        return int(out[0]), out[1], out[2]

    def act_batch(self, observations):
        """Same as act() for a batch [N, C, 64, 64] (several environments at once).
        Returns numpy arrays: actions [N] int64, log-probs [N], values [N]."""
        with torch.no_grad():
            states = torch.as_tensor(observations).to(self.device, non_blocking=True)
            dist, values = self._policy_value(states)
            values = self._denorm(values)
            actions = dist.sample()
            out = torch.stack([actions.float(), dist.log_prob(actions), values]).cpu().numpy()
        return out[0].astype(np.int64), out[1], out[2]

    def _advantages(self, reward_arr, values, dones_arr, last_value=None):
        """GAE: a_t = delta_t + gamma*lambda*a_{t+1}, delta_t = r_t + gamma*V(s_{t+1})*(1-done_t) - V(s_t).

        2022 behaviour (both fixes off): the last step gets 0 (no V(s_T) available) and the
        sum of deltas runs across episode boundaries.
        gae_cut_at_done: a_t = delta_t + gamma*lambda*(1-done_t)*a_{t+1} (standard GAE).
        bootstrap (last_value = V(s_T)): the last step also gets an advantage.
        """
        r = np.asarray(reward_arr, dtype=np.float64)
        v = np.asarray(values, dtype=np.float64)
        d = np.asarray(dones_arr, dtype=np.float64)
        adv = np.zeros(len(r), dtype=np.float64)
        n = len(r)
        if self.bootstrap and last_value is not None:
            v = np.append(v, float(last_value))   # V(s_T), the state after the last step
            first = n - 1
        else:
            first = n - 2
        a = 0.0
        for t in range(first, -1, -1):
            delta = r[t] + self.gamma * v[t + 1] * (1 - d[t]) - v[t]
            carry = (1 - d[t]) if self.gae_cut_at_done else 1.0
            a = delta + self.gamma * self.gae_lambda * carry * a
            adv[t] = a
        return adv.astype(np.float32)

    def value(self, observations):
        """Critic value(s) V(s) as numpy, for one observation [1, 4, 64, 64] or a batch."""
        with torch.no_grad():
            states = torch.as_tensor(observations).to(self.device)
            _, values = self._policy_value(states)
            return self._denorm(values).cpu().numpy().astype(np.float64)

    def learn(self, last_obs=None):
        """One PPO update on the rollout stored in self.memory (single environment).
        last_obs: observation after the last stored step (used only if bootstrap is on)."""
        _, action_arr, old_prob_arr, vals_arr, reward_arr, dones_arr, _ = self.memory.generate_batches()
        # Advantages (GAE). They depend only on the rollout, not on the epoch, so they are
        # computed once. The original code recomputed them at every epoch with a double loop
        # (O(n^2), ~2M iterations, ~20 s per rollout). This backward recursion gives exactly the
        # same numbers (checked bit-for-bit), including the original behaviour that the
        # discount chain is *not* cut at episode boundaries and the last step gets 0.
        last_value = self.value(last_obs)[0] if (self.bootstrap and last_obs is not None) else None
        advantage = self._advantages(reward_arr, vals_arr, dones_arr, last_value)
        self.update(self.memory.states, action_arr, old_prob_arr, vals_arr, advantage)
        self.memory.clear_memory()

    def update(self, state_arr, action_arr, old_prob_arr, vals_arr, advantage):
        """PPO epochs on a rollout given as flat arrays (any number of environments).

        Same computation as the 2022 code, including its dtypes (old log-probs and values in
        float64, advantages in float32). Only the data movement changed: the whole rollout is
        copied to the GPU once, instead of once per mini-batch (320 copies of 4 MB per rollout).
        """
        device = self.device
        n_states = len(state_arr)
        states_all = torch.as_tensor(state_arr).to(device)
        old_probs_all = torch.as_tensor(np.asarray(old_prob_arr, dtype=np.float64)).to(device)
        actions_all = torch.as_tensor(np.asarray(action_arr, dtype=np.int64)).to(device)
        advantage = torch.as_tensor(advantage).to(device)
        values = torch.as_tensor(np.asarray(vals_arr, dtype=np.float64)).to(device)
        if self.value_norm is not None:
            # update the running statistics with this rollout's value targets, then train the
            # critic on targets normalised with them
            self.value_norm.update((advantage.double() + values).cpu().numpy())
            vn_mean, vn_std = self.value_norm.stats()

        actor_losses = []
        critic_losses = []
        entropies = []

        for _ in range(self.n_epochs):
            # same shuffling as PPOMemory.generate_batches
            indices = np.arange(n_states, dtype=np.int64)
            np.random.shuffle(indices)
            batches = [indices[i:i + self.memory.batch_size] for i in range(0, n_states, self.memory.batch_size)]

            for batch in batches:
                batch = torch.as_tensor(batch).to(device)
                states = states_all[batch]
                old_probs = old_probs_all[batch]
                actions = actions_all[batch]

                dist, critic_value = self._policy_value(states)

                new_probs = dist.log_prob(actions)
                prob_ratio = new_probs.exp() / old_probs.exp()

                adv = advantage[batch]
                if self.norm_adv:
                    # zero mean, unit std in this mini-batch: the size of the policy step no
                    # longer depends on the scale of the rewards. Only for the policy loss: the
                    # critic still learns the raw returns below.
                    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
                weighted_probs = adv * prob_ratio
                weighted_clipped_probs = torch.clamp(prob_ratio, 1 - self.policy_clip,
                                                     1 + self.policy_clip) * adv
                actor_loss = -torch.min(weighted_probs, weighted_clipped_probs).mean()

                returns = advantage[batch] + values[batch]
                if self.value_norm is not None:
                    returns = (returns - vn_mean) / vn_std

                # Critic loss
                critic_loss = (returns - critic_value) ** 2
                critic_loss = critic_loss.mean()

                # Entropy of the policy on this mini-batch: high = actions spread out (exploring),
                # low = almost always the same action. Logged in every run, used in the loss
                # only when ent_coef > 0 (subtracted, so minimising the loss increases entropy).
                entropy = dist.entropy().mean()

                total_loss = actor_loss + 0.5 * critic_loss - self.ent_coef * entropy

                for net in self.networks:
                    net.optimizer.zero_grad()
                total_loss.backward()
                for net in self.networks:   # gradient norm clipped per network (2022: per network)
                    torch.nn.utils.clip_grad_norm_(net.parameters(), self.max_grad_norm)
                    net.optimizer.step()

                # logged values, kept on the GPU and read once at the end (no sync per mini-batch)
                actor_losses.append(actor_loss.detach())
                critic_losses.append(critic_loss.detach())
                entropies.append(entropy.detach())

        # Losses logs
        self.avg_critic_loss = torch.stack(critic_losses).mean().item()
        self.avg_actor_loss = torch.stack(actor_losses).mean().item()
        self.total_loss = total_loss.item()
        self.avg_entropy = torch.stack(entropies).mean().item()
