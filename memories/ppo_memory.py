import numpy as np
import torch


class PPOMemory:
    def __init__(self, batch_size, input_dims, n_rollouts):

        self.input_dims = input_dims

        # self.states = []
        self.probs = []
        self.vals = []
        self.actions = []
        self.rewards = []
        self.dones = []

        self.mem_size = n_rollouts

        self.states = np.zeros((self.mem_size, *input_dims), dtype=np.float32)

        self.batch_size = batch_size

        self.mem_count = 0


    def generate_batches(self):
        n_states = len(self.states)
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i+self.batch_size] for i in batch_start]

        return  self.states,\
                np.array(self.actions),\
                np.array(self.probs),\
                np.array(self.vals),\
                np.array(self.rewards),\
                np.array(self.dones),\
                batches

    def store_memory(self, state, action, probs, vals, reward, done):
        # self.states.append(state)
        self.actions.append(action)
        self.probs.append(probs)
        self.vals.append(vals)
        self.rewards.append(reward)
        self.dones.append(done)

        self.states[self.mem_count] = state # If index out of range, rollout isn't good

        self.mem_count += 1

    def clear_memory(self):
        self.states = np.zeros((self.mem_size, *self.input_dims), dtype=np.float32)

        self.probs = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.vals = []
        self.mem_count = 0