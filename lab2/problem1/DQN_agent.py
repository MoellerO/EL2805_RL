# Authors: Jing Li (19980512-9260) and Oliver Möller (19980831-2913)

# Load packages
import gym
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from collections import deque, namedtuple


class Agent(object):
    ''' Base agent class, used as a parent class
        Args:
            n_actions (int): number of actions
        Attributes:
            n_actions (int): where we store the number of actions
            last_action (int): last action taken by the agent
    '''
    def __init__(self, my_network):
        self.my_network = my_network

    def forward(self, state: np.ndarray):
        ''' Performs a forward computation '''
        network = torch.load(self.my_network)
        q_values = network(torch.tensor([state]))
        action = q_values.max(1)[1].item()
        return action

    def backward(self):
        ''' Performs a backward pass on the network '''
        pass


class RandomAgent(Agent):
    ''' Agent taking actions uniformly at random, child of the class Agent'''
    def __init__(self, n_actions: int):
        # super(RandomAgent, self).__init__(n_actions)
        self.n_actions = n_actions

    def forward(self, state: np.ndarray) -> int:
        ''' Compute an action uniformly at random across n_actions possible
            choices
            Returns:
                action (int): the random action
        '''
        self.last_action = np.random.randint(0, self.n_actions)
        return self.last_action


class my_NN(nn.Module):
    """ Create a feedforward neural network """
    def __init__(self, input_size, output_size, hidden_size=64):
        super().__init__()

        self.input_layer = nn.Linear(input_size, hidden_size)
        self.activate1 = nn.ReLU()

        self.hidden_layer = nn.Linear(hidden_size, hidden_size)
        self.activate2 = nn.ReLU()

        self.output_layer = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        l1 = self.input_layer(x)
        l1 = self.activate1(l1)

        l2 = self.hidden_layer(l1)
        l2 = self.activate2(l2)

        output = self.output_layer(l2)
        return output


Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])


class ExperienceReplayBuffer(object):
    """ Class used to store a buffer containing experiences of the RL agent.
    """
    def __init__(self, maximum_length):
        # Create buffer of maximum length
        self.buffer = deque(maxlen=maximum_length)

    def append(self, experience):
        # Append experience to the buffer
        self.buffer.append(experience)

    def __len__(self):
        # overload len operator
        return len(self.buffer)

    def sample_batch(self, n):
        """ Function used to sample experiences from the buffer.
            returns 5 lists, each of size n. Returns a list of state, actions,
            rewards, next states and done variables.
        """
        # If we try to sample more elements that what are available from the
        # buffer we raise an error
        if n > len(self.buffer):
            raise IndexError('Tried to sample too many elements from the buffer!')

        # Sample without replacement the indices of the experiences
        # np.random.choice takes 3 parameters: number of elements of the buffer,
        # number of elements to sample and replacement.
        indices = np.random.choice(
            len(self.buffer),
            size=n,
            replace=False
        )

        # Using the indices that we just sampled build a list of chosen experiences
        batch = [self.buffer[i] for i in indices]

        # batch is a list of size n, where each element is an Experience tuple
        # of 5 elements. To convert a list of tuples into
        # a tuple of list we do zip(*batch). In this case this will return a
        # tuple of 5 elements where each element is a list of n elements.
        #return batch #I have changed it for the starting point of the training
        return zip(*batch)


class DQNAgent():
    def __init__(self, batch_size, discount_factor, learning_rate, num_actions, dim_state):
        self.batch_size = batch_size
        self.discount_factor = discount_factor
        self.learning_rate = learning_rate
        self.num_actions = num_actions
        self.dim_state = dim_state

        # Create main network and target network
        self.main_network = my_NN(dim_state, num_actions)
        self.target_network = my_NN(dim_state, num_actions)

        # Set optimizer
        self.optimizer = optim.Adam(self.main_network.parameters(), lr=self.learning_rate)

    # Choose action based on epsilon greedy
    def forward(self, state, epsilon):
        rand_num = np.random.uniform(0, 1)
        if rand_num < epsilon:
            action = np.random.choice(self.num_actions)
        else:
            try:
                q_values = self.main_network(torch.tensor([state]))
            except:
                q_values = self.main_network(torch.tensor([state[0]]))
            action = q_values.max(1)[1].item()

        return action

    def train(self, buffer):
        # Sample experiences from the buffer
        states, actions, rewards, next_states, dones = buffer.sample_batch(n=self.batch_size)

        # Initialize the gradient to zero
        self.optimizer.zero_grad()

        output_target = self.target_network(torch.tensor(next_states, requires_grad=True, dtype=torch.float32))

        # Compute the target y values
        y_values = (torch.tensor(rewards, dtype=torch.float32)) + (self.discount_factor * output_target.max(1)[0]) \
                   * (torch.tensor(dones) == False)

        # Compute output
        states_list = list(states)
        for i in range(50):
            if len(states_list[i]) != 8:
                states_list[i] = states_list[i][0]
        states = tuple(states_list)
        q_values = self.main_network.forward(torch.tensor(states, requires_grad=True, dtype=torch.float32))\
            .gather(1, torch.tensor(actions).unsqueeze(1)).reshape(-1)

        # Compute loss
        loss = nn.functional.mse_loss(q_values, y_values)

        loss.backward()

        # Clip gradient
        nn.utils.clip_grad_norm_(self.main_network.parameters(), max_norm=1.)

        self.optimizer.step()

    def update_ann(self):
        self.target_network.load_state_dict(self.main_network.state_dict())

    def save_ann(self, filename_main='neural-network-1.pth'):
        torch.save(self.main_network, filename_main)

