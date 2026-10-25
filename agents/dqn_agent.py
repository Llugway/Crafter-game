import torch
import numpy as np
from networks.q_network import DeepQNetwork


class DQN_Agent():

    def __init__(self, gamma, lr, input_dims, batch_size, n_actions,
                 max_mem_size=100000, epsilon=1.0, eps_end=0.01, eps_dec=5e-4):

        self.loss_log = []
        self.q_values_log = []

        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_min = eps_end
        self.eps_dec = eps_dec
        self.lr = lr
        self.action_space = [i for i in range(n_actions)]
        self.mem_size = max_mem_size
        self.batch_size = batch_size
        self.mem_cntr = 0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # self.replayMemory = ReplayMemory(size=max_mem_size,batch_size=batch_size)
        self.Q_eval = DeepQNetwork(self.lr, n_actions=n_actions, input_dims=input_dims).to(self.device)

        self.state_memory = np.zeros((self.mem_size, *input_dims), dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, *input_dims), dtype=np.float32)
        self.action_memory = np.zeros((self.mem_size), dtype=np.int32)
        self.reward_memory = np.zeros((self.mem_size), dtype=np.float32)
        self.terminal_memory = np.zeros((self.mem_size), dtype=np.bool)

    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size

        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done

        self.mem_cntr += 1

    def act(self, observation, train=None):
        if np.random.random() > self.epsilon:
            state = observation.clone().detach()
            state = state.to(self.device)
            actions = self.Q_eval.forward(state)
            q_value = torch.max(actions).item()
            self.q_values_log.append(q_value)
            action = torch.argmax(actions).item()
        else:
            action = np.random.choice(self.action_space)

        return action

    def learn(self):
        if self.mem_cntr < self.batch_size:
            return

        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, self.batch_size, replace=False)  # or True

        batch_index = np.arange(self.batch_size, dtype=np.int32)

        state_batch = torch.tensor(self.state_memory[batch]).to(self.device)
        new_state_batch = torch.tensor(self.new_state_memory[batch]).to(self.device)
        reward_batch = torch.tensor(self.reward_memory[batch]).to(self.device)
        terminal_batch = torch.tensor(1 * self.terminal_memory[batch]).to(self.device)
        action_batch = self.action_memory[batch]

        q_values = self.Q_eval.forward(state_batch)
        q_value_next = self.Q_eval.forward(new_state_batch)

        predicted_value_of_now = q_values[batch_index, action_batch]
        predicted_value_of_future = torch.max(q_value_next, dim=1)[0]

        q_target = reward_batch + self.gamma * predicted_value_of_future * (1 - terminal_batch)

        loss = self.Q_eval.loss(q_target, predicted_value_of_now).to(self.device)

        self.loss_log.append(loss.item())

        self.Q_eval.optimizer.zero_grad()
        loss.backward()
        self.Q_eval.optimizer.step()

        self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_min \
            else self.eps_min

    def remove_logs(self):
        self.loss_log = []
        self.q_values_log = []