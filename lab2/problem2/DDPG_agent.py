# Authors: Jing Li (19980512-9260) and Oliver Möller (19980831-2913)

# Copyright [2020] [KTH Royal Institute of Technology] Licensed under the
# Educational Community License, Version 2.0 (the "License"); you may
# not use this file except in compliance with the License. You may
# obtain a copy of the License at http://www.osedu.org/licenses/ECL-2.0
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS"
# BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied. See the License for the specific language governing
# permissions and limitations under the License.
#
# Course: EL2805 - Reinforcement Learning - Lab 2 Problem 2
# Code author: [Alessio Russo - alessior@kth.se]
# Last update: 26th October 2020, by alessior@kth.se
#

# Load packages
import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from collections import deque


class Actor(nn.Module):
    ''' Actor network. Gets an state and outputs an action.
    '''

    def __init__(self, device):
        super(Actor, self).__init__()
        self.input_dim = 8
        self.fc1_dim = 400
        self.fc2_dim = 200
        self.out_dim = 2
        lr = 5e-4

        self.fc1 = nn.Linear(self.input_dim, self.fc1_dim)
        self.fc2 = nn.Linear(self.fc1_dim, self.fc2_dim)
        self.fc3 = nn.Linear(self.fc2_dim, self.out_dim)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        # self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(device)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        action = F.tanh(self.fc3(x))
        return action


class Critic(nn.Module):
    ''' Critic modeling pi. Takes state and action, returns value.
    '''

    def __init__(self, device):
        super(Critic, self).__init__()
        self.input_dim = 8
        self.fc1_dim = 400
        self.fc2_dim = 200
        self.nmb_actions = 2
        lr = 5e-3

        self.fc1 = nn.Linear(self.input_dim, self.fc1_dim)
        self.a1 = nn.PReLU()
        self.fc2 = nn.Linear(self.fc1_dim + self.nmb_actions, self.fc2_dim)
        self.a2 = nn.PReLU()
        self.fc3 = nn.Linear(self.fc2_dim, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.to(device)

    def forward(self, state, action):
        x = self.a1(self.fc1(state))
        concat_action = T.cat([x, action], dim=1)
        x = self.a2(self.fc2(concat_action))
        out = self.fc3(x)
        return out


class RandomAgent(object):
    ''' Agent taking actions uniformly at random, child of the class Agent'''

    def __init__(self, n_actions: int):
        super(RandomAgent, self).__init__(n_actions)

    def forward(self, state: np.ndarray) -> np.ndarray:
        ''' Compute a random action in [-1, 1]

            Returns:
                action (np.ndarray): array of float values containing the
                    action. The dimensionality is equal to self.n_actions from
                    the parent class Agent.
        '''
        return np.clip(-1 + 2 * np.random.rand(self.n_actions), -1, 1)
