import random
import numpy as np
import torch
from torch import optim
import torch.nn.functional as F
from networks.q_network import DeepQNetwork


class M_DQN_Agent():
    """Interacts with and learns from the environment."""

    def __init__(self,
                 state_size,
                 action_size,
                 layer_size,
                 BATCH_SIZE,
                 BUFFER_SIZE,
                 LR,
                 TAU,
                 GAMMA,
                 UPDATE_EVERY,
                 device,
                 entropy_tau=0.03,
                 alpha=0.9,
                 lo=-1
                 ):
        """Initialize an Agent object.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            layer_size (int): size of the hidden layer
            BATCH_SIZE (int): size of the training batch
            BUFFER_SIZE (int): size of the replay memory
            LR (float): learning rate
            TAU (float): tau for soft updating the network weights
            GAMMA (float): discount factor
            UPDATE_EVERY (int): update frequency
            device (str): device that is used for the compute
            seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.device = device
        self.TAU = TAU
        self.GAMMA = GAMMA
        self.UPDATE_EVERY = UPDATE_EVERY
        self.BATCH_SIZE = BATCH_SIZE
        self.Q_updates = 0
        self.entropy_tau = entropy_tau
        self.alpha = alpha
        self.lo = lo

        self.action_step = 4
        self.last_action = None

        # Q-Network
        self.qnetwork_local = DeepQNetwork(LR, state_size, action_size).to(device)
        self.qnetwork_target = DeepQNetwork(LR, state_size, action_size).to(device)

        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

        # Replay memory
        self.memory = ReplayBuffer(BUFFER_SIZE, BATCH_SIZE, self.device, self.GAMMA, 1)

        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0

    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)

        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % self.UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > self.BATCH_SIZE:
                experiences = self.memory.sample()
                loss = self.learn(experiences)
                self.Q_updates += 1
                # writer.add_scalar("Q_loss", loss, self.Q_updates)

    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy. Acting only every 4 frames!

        Params
        ======
            frame: to adjust epsilon
            state (array_like): current state

        """

        if self.action_step == 4:
            state = np.array(state)

            state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
            self.qnetwork_local.eval()
            with torch.no_grad():
                action_values = self.qnetwork_local(state)
            self.qnetwork_local.train()

            # Epsilon-greedy action selection
            if random.random() > eps: # select greedy action if random number is higher than epsilon or noisy network is used!
                action = np.argmax(action_values.cpu().data.numpy())
                self.last_action = action
                return action
            else:
                action = random.choice(np.arange(self.action_size))
                self.last_action = action
                return action
            # self.action_step = 0
        else:
            self.action_step += 1
            return self.last_action

    def learn(self, experiences):
        """Update value parameters using given batch of experience tuples.
        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        self.optimizer.zero_grad()
        states, actions, rewards, next_states, dones = experiences
        # Get predicted Q values (for next states) from target model
        Q_targets_next = self.qnetwork_target(next_states).detach()
        # calculate entropy term with logsum
        logsum = torch.logsumexp( \
            (Q_targets_next - Q_targets_next.max(1)[0].unsqueeze(-1) ) /self.entropy_tau , 1).unsqueeze(-1)

        tau_log_pi_next = Q_targets_next - Q_targets_next.max(1)[0].unsqueeze(-1) - self.entropy_tau *logsum
        # target policy
        pi_target = F.softmax(Q_targets_next /self.entropy_tau, dim=1)
        Q_target = (self.GAMMA * (pi_target * (Q_targets_next - tau_log_pi_next ) * (1 - dones)).sum(1)).unsqueeze(-1)

        # calculate munchausen addon with logsum trick
        q_k_targets = self.qnetwork_target(states).detach()
        v_k_target = q_k_targets.max(1)[0].unsqueeze(-1)
        logsum = torch.logsumexp((q_k_targets - v_k_target ) /self.entropy_tau, 1).unsqueeze(-1)
        log_pi = q_k_targets - v_k_target - self.entropy_tau *logsum
        munchausen_addon = log_pi.gather(1, actions)

        # calc munchausen reward:
        munchausen_reward = (rewards + self.alpha*torch.clamp(munchausen_addon, min=self.lo, max=0))

        # Compute Q targets for current states
        Q_targets = munchausen_reward + Q_target

        # Get expected Q values from local model
        q_k = self.qnetwork_local(states)
        Q_expected = q_k.gather(1, actions)

        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets)  # mse_loss
        # Minimize the loss
        loss.backward()
        # clip_grad_norm_(self.qnetwork_local.parameters(),1)
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target)
        return loss.detach().cpu().numpy()

    def soft_update(self, local_model, target_model):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.TAU *local_param.data + (1.0 - self.TAU ) *target_param.data)