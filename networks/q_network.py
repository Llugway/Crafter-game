from torch import nn
import torch.optim as optim
import torch.nn.functional as F

class DeepQNetwork(nn.Module):
    def __init__(self, lr, input_dims, n_actions):
        super(DeepQNetwork, self).__init__()
        self.input_dims = input_dims
        self.n_actions = n_actions

        self.conv1 = nn.Conv2d(in_channels=input_dims[0], out_channels=32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)

        self.relu = nn.ReLU()

        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, self.n_actions)


        # self.optimizer = optim.RMSprop(self.parameters(), lr=lr)

        # self.loss = nn.SmoothL1Loss()


    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        actions = self.fc2(x)
        return actions