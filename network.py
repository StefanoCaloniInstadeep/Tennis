import torch
import torch.nn as nn
import torch.nn.functional as F


class CriticNetwork(nn.Module):
    def __init__(self, state_size, action_size, fc1_size=64, fc2_size=64):
        super(CriticNetwork, self).__init__()

        self.fc1 = nn.Linear(state_size+action_size, fc1_size)
        self.fc2 = nn.Linear(fc1_size, fc2_size)
        self.fc3 = nn.Linear(fc2_size, 1)

    def forward(self, state, action):
        x = torch.cat((state, action), dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class ActorNetwork(nn.Module):
    def __init__(self, state_size, action_size, fc1_size=64, fc2_size=64):
        super(ActorNetwork, self).__init__()

        self.fc1 = nn.Linear(state_size, fc1_size)
        self.fc2 = nn.Linear(fc1_size, fc2_size)
        self.fc3 = nn.Linear(fc2_size, action_size)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        action_mean = F.tanh(self.fc3(x))

        return action_mean
