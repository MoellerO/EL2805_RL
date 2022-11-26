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
# Course: EL2805 - Reinforcement Learning - Lab 1 Problem 4
# Code author: [Alessio Russo - alessior@kth.se]
# Last update: 6th October 2020, by alessior@kth.se
#

# Load packages
import numpy as np
import gym
import torch
import matplotlib.pyplot as plt
import matplotlib
import pickle
import random

# Import and initialize Mountain Car Environment
env = gym.make('MountainCar-v0')
env.reset()
k = env.action_space.n      # tells you the number of actions
low, high = env.observation_space.low, env.observation_space.high

random.seed(0)
np.random.seed(0)
# Parameters
N_episodes = 500      # Number of episodes to run for training
discount_factor = 1.   # Value of gamma
LAMBDA = 0.
ALPHA = 0.2
M = 0.2
P = 2  # Order of fourier base
ETA = np.array([[i, j] for i in range(P + 1) for j in range(P + 1)])
ETA = np.delete(ETA, (0), axis=0)  # remove [0,0]

EPS = 1.
w = np.zeros((ETA.shape[0], k))  # init weights
v = np.zeros((ETA.shape[0], k))  # init weights
Q_w = np.zeros((k))

# Reward
episode_reward_list = []  # Used to save episodes reward


# Functions used during training
def running_average(x, N):
    ''' Function used to compute the running mean
        of the last N elements of a vector x
    '''
    if len(x) >= N:
        y = np.copy(x)
        y[N - 1:] = np.convolve(x, np.ones((N, )) / N, mode='valid')
    else:
        y = np.zeros_like(x)
    return y


def scale_state_variables(s, low=env.observation_space.low, high=env.observation_space.high):
    ''' Rescaling of s to the box [0,1]^2 '''
    x = (s - low) / (high - low)
    return x


# own functions

def phi(s, p=2):
    p = 2
    return np.cos(np.pi * np.matmul(ETA, s))


def choose_action(eps, Q):
    if np.random.random() < eps:
        return np.random.choice(np.arange(k))
    else:
        return np.argmax(Q)


def scale_alpha(alpha):
    eta_normalitzation = np.linalg.norm(ETA, 2)
    if eta_normalitzation == 0:
        eta_normalitzation = 1
    return alpha / eta_normalitzation


def decay_alpha(alpha):
    if total_episode_reward > -130:
        if alpha > 0.000001:
            alpha *= 0.8
    elif total_episode_reward < -160:
        if alpha < 0.1:
            alpha = alpha / 0.8
    print(total_episode_reward)
    return scale_alpha(alpha)

    # if i % 10 == 0:
    #     alpha = alpha * 0.9
    # if total_episode_reward > -100:
    #     alpha *= 1 - 0.2 * (total_episode_reward > -100)
    # elif total_episode_reward > -130:
    #     alpha *= 1 - 0.1 * (total_episode_reward > -130)
    # elif total_episode_reward > -150:
    #     alpha *= 1 - 0.01 * (total_episode_reward > -150)


print(phi(scale_state_variables(env.reset())))

if 1:
    alpha = scale_alpha(ALPHA)
    eps = EPS
    # Training process
    for i in range(N_episodes):
        # Reset enviroment data
        done = False
        total_episode_reward = 0.

        # init state and action
        state = scale_state_variables(env.reset())
        a_t = choose_action(eps, Q_w)

        # reset eligibility trace and velocity term used for momentum
        z = np.zeros((ETA.shape[0], k))
        v = np.zeros((ETA.shape[0], k))
        while not done:
            # env.action_space.n tells you the number of actions
            # available

            # Get next state and reward.  The done variable
            # will be True if you reached the goal position,
            # False otherwise
            next_state, reward, done, _ = env.step(a_t)
            next_state = scale_state_variables(next_state)
            action_next = choose_action(0, Q_w)

            # Update episode reward
            total_episode_reward += reward
            delta_t = reward + discount_factor * \
                np.dot(w[:, action_next].T, phi(next_state)) - \
                np.dot(w[:, a_t].T, phi(state))

            # Update z
            for a in range(k):
                if a == a_t:
                    # note that the derivative of our approx. function w_a^T*phi(s) is simply phi(s)
                    z[:, a] = discount_factor * LAMBDA * z[:, a] + phi(state)
                else:
                    z[:, a] = discount_factor * LAMBDA * z[:, a]

            z = np.clip(z, -5, 5)

            # Update w with SGD with Nesterov Acceleration
            v = M * v + alpha * delta_t * z
            w = w + M * v + alpha * delta_t * z
            # w = w + v # only momentum

            # Update Q_w
            Q_w = np.dot(w.T, phi(state))

            # Update state for next iteration
            state = next_state
            a_t = action_next

        alpha = decay_alpha(alpha)
        print(i, 'alpha', alpha)
        # Append episode reward
        episode_reward_list.append(total_episode_reward)
        if eps > 0.01:
            eps = eps * 0.99

        # Close environment
        env.close()
    with open('weights.pkl', 'wb') as handle:
        N = ETA
        data = {'W': w.T, 'N': N}
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # Plot Rewards
    plt.plot([i for i in range(1, N_episodes + 1)],
             episode_reward_list, label='Episode reward')
    plt.plot([i for i in range(1, N_episodes + 1)],
             running_average(episode_reward_list, 10), label='Average episode reward')
    plt.xlabel('Episodes')
    plt.ylabel('Total reward')
    plt.title('Total Reward vs Episodes')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()

if 1:
    with open('weights.pkl', 'rb') as f:
        data = pickle.load(f)
    w = data['W'].T
    # Plot optimal val func
    s0 = np.linspace(0, 1, 100)
    s1 = np.linspace(0, 1, 100)
    X, Y = np.meshgrid(s0, s1)
    Z = np.array([[max(np.dot(w.T, phi(np.array([p, v]))))
                   for p in s0] for v in s1])
    fig, ax = plt.subplots()
    surf = ax.pcolormesh(X, Y, Z, shading='auto')
    plt.xlabel('s1 - Position')
    plt.ylabel('s2 - Velocity')
    plt.title('Value of Optimal Policy')
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()

    # Plot optimal policy
    s0 = np.linspace(0, 1, 100)
    s1 = np.linspace(0, 1, 100)
    X, Y = np.meshgrid(s0, s1)
    Z = np.array([[np.argmax(np.dot(w.T, phi(np.array([p, v]))))
                 for p in s0] for v in s1])
    fig, ax = plt.subplots()
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        "", ["white", "grey", "black"])
    surf = ax.pcolormesh(X, Y, Z, cmap=cmap, shading='auto')
    plt.xlabel('Position')
    plt.ylabel('Velocity')
    plt.title('Optimal Policy by State')
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()
