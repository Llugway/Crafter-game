"""Achievement Distillation, intra-trajectory achievement prediction (Moon et al., NeurIPS 2023).

The idea: PPO only learns from the reward, which in Crafter is a +1 every now and then. The
agent's encoder is never asked to understand *what* it is working towards. Achievement
Distillation adds a self-supervised task, trained in a separate "auxiliary phase" after every
few PPO updates (as in Phasic Policy Gradient):

    for each step t, recognise which achievement will be unlocked next.

  - The next achievement g+ is represented by the change it causes in the encoder:
        nu(g+) = normalise(phi(s_{u+1}) - phi(s_u))     u = step where it is unlocked
  - The step itself is represented by psi(s_t, a_t): the encoder features phi(s_t), modulated
    by the action taken (FiLM: (1 + eta(a)) * phi + delta(a)), then a 2-layer MLP, normalised.
  - Contrastive loss: psi(s_t, a_t) must be closer to nu(g+) than psi(s_t', a_t') of a random
    other step t' of the same episode:
        L_pred = -log( exp(psi_t . nu / T) / (exp(psi_t . nu / T) + exp(psi_t' . nu / T)) )

To keep the policy and value from drifting while the shared encoder is trained on this task,
two regularisers tie them to their state before the auxiliary phase (pi_old, V_old):
        R_pi = KL(pi_old(.|s) || pi(.|s)),   R_V = 1/2 (V(s) - V_old(s))^2,   both with weight 1.

Implemented here: this intra-trajectory prediction, which gives the largest gain in the
paper's ablation (PPO 15.6% -> 19.0% at 1M steps). Not (yet) implemented: cross-trajectory
achievement matching and the achievement memory (-> 20.4% and 21.8%).

Achievements are detected from the reward: an unlock gives +1, health changes only +-0.1 per point
(threshold 0.5, so an unlock on the same step as a -0.1 health tick still counts).
"""
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


# --------------------------------------------------------------- indices (numpy only) ---
def achievement_targets(rewards, dones, threshold=0.5):
    """For one environment's sequence of steps (rewards [L], dones [L]).

    Returns, for each step t:
      next_ach[t]  step u >= t of the next achievement unlock in the same episode, -1 if none
                   visible in the buffer (then t is not used as an anchor);
      seg_start[t], seg_end[t]  first and last step of t's episode segment in the buffer
                   (to draw the negative t' from the same episode).
    An unlock on the very last step of an episode is ignored: with automatic resets the next
    observation already belongs to the next episode, so phi(s_{u+1}) is not available.
    """
    rewards = np.asarray(rewards)
    dones = np.asarray(dones, dtype=bool)
    n = len(rewards)
    unlock = (rewards >= threshold) & ~dones
    next_ach = np.full(n, -1, dtype=np.int64)
    seg_end = np.zeros(n, dtype=np.int64)
    nxt, end = -1, n - 1
    for t in range(n - 1, -1, -1):
        if dones[t]:              # step t is the last step of its episode
            nxt, end = -1, t
        if unlock[t]:
            nxt = t
        next_ach[t] = nxt
        seg_end[t] = end
    seg_start = np.zeros(n, dtype=np.int64)
    start = 0
    for t in range(n):
        seg_start[t] = start
        if dones[t]:
            start = t + 1
    return next_ach, seg_start, seg_end


# ---------------------------------------------------------------------- the module ---
class AchievementDistillation(nn.Module):
    def __init__(self, model, n_actions=17, dim=1024, hidden=1024, lr=3e-4, temperature=0.1,
                 beta_pi=1.0, beta_v=1.0, epochs=6, batch_size=256, chunk=64, max_grad_norm=0.5):
        super().__init__()
        self.model = model                      # the shared ImpalaActorCritic
        self.n_actions = n_actions
        self.temperature, self.beta_pi, self.beta_v = temperature, beta_pi, beta_v
        self.epochs, self.batch_size, self.max_grad_norm = epochs, batch_size, max_grad_norm
        self.chunk = chunk   # anchors per forward/backward pass (memory), see auxiliary_phase
        # FiLM action modulation and the psi head (2-layer MLPs, hidden size 1024)
        self.eta = nn.Sequential(nn.Linear(n_actions, hidden), nn.ReLU(), nn.Linear(hidden, dim))
        self.delta = nn.Sequential(nn.Linear(n_actions, hidden), nn.ReLU(), nn.Linear(hidden, dim))
        self.psi_head = nn.Sequential(nn.Linear(dim, hidden), nn.ReLU(), nn.Linear(hidden, dim))
        self.device = model.device
        self.to(self.device)
        # one optimiser for the auxiliary phase: encoder + heads + the new modules
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.last_stats = {}

    def psi(self, features, actions):
        a = F.one_hot(actions, self.n_actions).float()
        x = (1 + self.eta(a)) * features + self.delta(a)
        return F.normalize(self.psi_head(x), dim=-1)

    @staticmethod
    def _to_float(obs_uint8):
        return obs_uint8.float().div_(255.0)

    def auxiliary_phase(self, obs, next_obs, actions, rewards, dones):
        """obs uint8 [L, N, C, 64, 64] (L steps, N environments), next_obs [N, C, 64, 64] (after
        the last step), actions/rewards/dones [L, N]. Runs `epochs` passes of L_pred + R_pi + R_V."""
        L, N = actions.shape
        dev = self.device
        # ---- targets, per environment column, flattened to index i = t * N + env
        nxt = np.full((L, N), -1, dtype=np.int64)
        s0 = np.zeros((L, N), dtype=np.int64)
        s1 = np.zeros((L, N), dtype=np.int64)
        for e in range(N):
            nxt[:, e], s0[:, e], s1[:, e] = achievement_targets(rewards[:, e], dones[:, e])
        valid_t, valid_e = np.nonzero(nxt >= 0)
        if len(valid_t) == 0:
            self.last_stats = {"aux_loss": float("nan"), "pred_acc": float("nan"), "anchors": 0}
            return
        # all observations, plus the one after the last step, on the GPU as uint8 (~0.5 GB)
        all_obs = torch.as_tensor(np.concatenate([obs, next_obs[None]], 0)).to(dev)   # [L+1, N, C, H, W]
        acts = torch.as_tensor(actions).to(dev)

        # ---- pi_old / V_old: the policy and value before this phase, on every buffered state
        with torch.no_grad():
            old_logits = torch.empty(L, N, self.n_actions, device=dev)
            old_values = torch.empty(L, N, device=dev)
            flat = all_obs[:L].reshape(L * N, *all_obs.shape[2:])
            for i in range(0, L * N, 1024):
                feats = self.model.encode(self._to_float(flat[i:i + 1024]))
                dist, v = self.model.heads(feats)
                old_logits.view(L * N, -1)[i:i + 1024] = dist.logits
                old_values.view(-1)[i:i + 1024] = v

        rng = np.random.default_rng()
        losses, accs, kls = [], [], []
        for _ in range(self.epochs):
            order = rng.permutation(len(valid_t))
            for b in range(0, len(order), self.batch_size):
                idx = order[b:b + self.batch_size]
                self.optimizer.zero_grad()
                # the batch is processed in chunks of anchors, gradients added up: same update as
                # one big batch, but the GPU only holds the activations of one chunk at a time
                for c in range(0, len(idx), self.chunk):
                    l_pred, r_pi, acc = self._chunk_loss(idx[c:c + self.chunk], len(idx), valid_t, valid_e,
                                                         nxt, s0, s1, all_obs, acts, old_logits, old_values, rng)
                    losses.append(l_pred)
                    accs.append(acc)
                    kls.append(r_pi)
                nn.utils.clip_grad_norm_(self.parameters(), self.max_grad_norm)
                self.optimizer.step()
        self.last_stats = {"aux_loss": torch.stack(losses).mean().item(),
                           "pred_acc": torch.stack(accs).mean().item(),
                           "kl_old": torch.stack(kls).mean().item(),
                           "anchors": int(len(valid_t))}
        # give the memory back: other runs share the GPU
        del all_obs, acts, old_logits, old_values
        if dev.type == "cuda":
            torch.cuda.empty_cache()

    def _chunk_loss(self, idx, batch_len, valid_t, valid_e, nxt, s0, s1, all_obs, acts, old_logits, old_values, rng):
        """Loss of a chunk of anchors, backpropagated with weight len(chunk) / batch_len."""
        dev = self.device
        t, e = valid_t[idx], valid_e[idx]
        u = nxt[t, e]                                         # step of the next unlock
        t_neg = rng.integers(s0[t, e], s1[t, e] + 1)          # random step, same episode
        # many anchors head to the same unlock: encode each unlock (s_u, s_u+1) only once
        unlocks, inverse = np.unique(np.stack([u, e], 1), axis=0, return_inverse=True)
        inverse = inverse.reshape(-1)
        tt, ee, nn_ = (torch.as_tensor(x, device=dev) for x in (t, e, t_neg))
        uu, ue = (torch.as_tensor(unlocks[:, k], device=dev) for k in (0, 1))
        B, U = len(idx), len(unlocks)
        batch = torch.cat([all_obs[tt, ee], all_obs[nn_, ee], all_obs[uu, ue], all_obs[uu + 1, ue]])
        feats = self.model.encode(self._to_float(batch))
        f_t, f_neg, f_u, f_u1 = feats[:B], feats[B:2 * B], feats[2 * B:2 * B + U], feats[2 * B + U:]

        nu = F.normalize(f_u1 - f_u, dim=-1)[torch.as_tensor(inverse, device=dev)]   # next-achievement embedding
        psi_pos = self.psi(f_t, acts[tt, ee])
        psi_neg = self.psi(f_neg, acts[nn_, ee])
        logits = torch.stack([(psi_pos * nu).sum(-1), (psi_neg * nu).sum(-1)], 1) / self.temperature
        l_pred = F.cross_entropy(logits, torch.zeros(B, dtype=torch.long, device=dev))

        # regularisers on the anchor states: stay close to pi_old and V_old
        dist, v = self.model.heads(f_t)
        old = torch.log_softmax(old_logits[tt, ee], -1)
        r_pi = (old.exp() * (old - dist.logits.log_softmax(-1))).sum(-1).mean()
        r_v = 0.5 * ((v - old_values[tt, ee]) ** 2).mean()

        loss = l_pred + self.beta_pi * r_pi + self.beta_v * r_v
        (loss * B / batch_len).backward()
        return l_pred.detach(), r_pi.detach(), (logits[:, 0] > logits[:, 1]).float().mean().detach()
