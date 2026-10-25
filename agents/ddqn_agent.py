from networks.q_network import DeepQNetwork
import torch
import numpy as np


class DDQN_Agent():
    """My own Agent"""

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

        self.Q1 = DeepQNetwork(self.lr, n_actions=n_actions, input_dims=input_dims).to(self.device)
        self.Q2 = DeepQNetwork(self.lr, n_actions=n_actions, input_dims=input_dims).to(self.device)

        for param in self.Q2.parameters():
            param.requires_grad = False

        self.Q2.load_state_dict(self.Q1.state_dict())

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

    def act(self, observation, train=True):
        if np.random.random() > self.epsilon:
            if (train):
                state = observation.clone().detach()
                state = state.to(self.device)
                actions = self.Q2.forward(state)

                q_value = torch.max(actions).item()
                self.q_values_log.append(q_value)

                action = torch.argmax(actions).item()
            else:
                state = observation.clone().detach()
                state = state.to(self.device)
                actions = self.Q1.forward(state)

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

        inter = self.state_memory[batch]

        state_batch = torch.tensor(inter).to(self.device)
        new_state_batch = torch.tensor(self.new_state_memory[batch]).to(self.device)
        reward_batch = torch.tensor(self.reward_memory[batch]).to(self.device)
        terminal_batch = torch.tensor(self.terminal_memory[batch]).to(self.device)
        action_batch = self.action_memory[batch]

        q_values = self.Q1.forward(state_batch)
        q_value_next = self.Q2.forward(new_state_batch)

        predicted_value_of_now = q_values[batch_index, action_batch]
        predicted_value_of_future = torch.max(q_value_next, dim=1)[0]

        q_target = reward_batch + self.gamma * predicted_value_of_future * terminal_batch

        loss = (predicted_value_of_now - q_target.detach()).pow(2).mean()

        self.loss_log.append(loss.item())

        self.Q1.optimizer.zero_grad()
        loss.backward()
        self.Q1.optimizer.step()

        self.Q2.load_state_dict(self.Q1.state_dict())

        self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_min \
            else self.eps_min

    def remove_logs(self):
        self.loss_log = []
        self.q_values_log = []