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
# Last update: 20th November 2020, by alessior@kth.se
#

# Load packages
from collections import namedtuple
from collections import deque


import numpy as np
import gym
import torch as T
import matplotlib.pyplot as plt
from tqdm import trange
from DDPG_agent import RandomAgent, Critic, Actor
import torch.optim as optim
import torch.nn as nn
from DDPG_soft_updates import soft_updates
from torchsummary import summary


# Parameters
N_episodes = 500               # Number of episodes to run for training
discount_factor = 0.99        # Value of gamma
n_ep_running_average = 50      # Running average of 50 episodes
L = 10000
N = 64
d = 2
tau = 1e-3
save_models = False
max_avg_reward = -100  # min treshold to save models (not in use)


def create_buffer(env, size, tuple_format):
    buffer = deque(maxlen=size)
    state = env.reset()
    cntr = 0
    while cntr < size:
        action = np.clip(-1 + 2 * np.random.rand(2), -1, 1)
        next_state, reward, done, _ = env.step(action)
        exp = tuple_format(state, action, reward, next_state, done)
        buffer.append(exp)
        cntr += 1
        if done:
            state = env.reset()
    return buffer


def get_batch(buffer, n):
    if n > len(buffer):
        raise IndexError(
            'Buffer not filled correctly.')
    idcs = np.random.choice(len(buffer), size=n, replace=False)
    batch = [buffer[i] for i in idcs]
    return zip(*batch)


def running_average(x, N):
    ''' Function used to compute the running average
        of the last N elements of a vector x
    '''
    if len(x) >= N:
        y = np.copy(x)
        y[N - 1:] = np.convolve(x, np.ones((N, )) / N, mode='valid')
    else:
        y = np.zeros_like(x)
    return y


def noise(eta_prev, mu=0.15, sigma=0.2):
    return -mu * eta_prev + np.random.multivariate_normal([0, 0], sigma**2 * np.eye(2), size=1)


# Import and initialize Lunar Lander Environment
env = gym.make('LunarLanderContinuous-v2')
env.reset()


# Agent initialization
dev = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
critic = Critic(dev)
target_critic = Critic(dev)
actor = Actor(dev)
target_actor = Actor(dev)
print(summary(actor, (8,)))
print(summary(critic, [(8,), (2,)]))

# Training process
EPISODES = trange(N_episodes, desc='Episode: ', leave=True)
EXPERIENCE = namedtuple('Experience',
                        ['state', 'action', 'reward', 'next_state', 'done'])
episode_reward_list = []  # Used to save episodes reward
episode_number_of_steps = []
buffer = create_buffer(env=env, size=L, tuple_format=EXPERIENCE)
for i in EPISODES:
    # Reset enviroment data
    done = False
    state = env.reset()
    total_episode_reward = 0.
    t = 0
    eta_prev = np.array([0, 0])
    while not done:

        eta = noise(eta_prev)
        with T.no_grad():
            act_state_batch = T.tensor(
                [state], requires_grad=False, dtype=T.float32).to(dev)
            action = actor.forward(act_state_batch)

        eta_prev = eta

        # Take a random action
        # action = agent.forward(state)
        action = action.detach().cpu().numpy().reshape(2,) + eta.reshape(2,).clip(-1, 1)

        # Get next state and reward.  The done variable
        # will be True if you reached the goal position,
        # False otherwise
        next_state, reward, done, _ = env.step(action)

        exp = EXPERIENCE(state, action, reward, next_state, done)
        buffer.append(exp)
        states, actions, rewards, next_states, dones = get_batch(buffer, N)

        # Training process, set gradients to 0
        critic.optimizer.zero_grad()

        # Compute output of the network given the state batch
        state_batch = T.tensor(states,
                               requires_grad=True, dtype=T.float32).to(dev)
        action_batch = T.tensor(
            actions, requires_grad=False, dtype=T.float32).detach().to(dev)
        Q_critic = critic.forward(
            state_batch, action_batch).view(N, 1)

        next_state_batch = T.tensor(
            next_states, requires_grad=False, dtype=T.float32).detach().to(dev)
        target_action = target_actor.forward(next_state_batch)
        target_action_batch = T.tensor(
            target_action, requires_grad=False, dtype=T.float32).detach().to(dev)
        Q_target = target_critic(
            next_state_batch, target_action_batch).max(1)[0]

        reward_batch = T.tensor(
            rewards, device=dev, dtype=T.float32).to(dev)
        dones_batch = T.tensor(list(map(int, dones))).to(dev)
        Q_values = (reward_batch + discount_factor * Q_target *
                    (1 - dones_batch)).view(N, 1)

        # train critic network
        critic.train(mode=True)
        # compute loss
        loss_critic = critic.loss(Q_critic, Q_values)
        # compute gradients
        loss_critic.backward()
        # clip gradients
        nn.utils.clip_grad_norm(critic.parameters(), max_norm=1.)
        # backpropagate
        critic.optimizer.step()

        # train actor network
        if t % d == 0:
            actor.optimizer.zero_grad()
            actor.train(mode=True)

            # compute loss
            state_batch_dt = T.tensor(
                states, requires_grad=False, dtype=T.float32).detach().to(dev)
            state_batch_gd = T.tensor(
                states, requires_grad=True, dtype=T.float32).to(dev)
            pred_action = actor(state_batch_gd)
            pred_val = critic(state_batch_dt, pred_action)

            loss_actor = -T.mean(pred_val)

            # compute gradients
            loss_actor.backward()
            # clip gradients
            nn.utils.clip_grad_norm(
                actor.parameters(), max_norm=1.)
            # backpropagate
            actor.optimizer.step()

            # update target networks
            target_actor = soft_updates(
                actor, target_actor, tau)
            target_critic = soft_updates(
                critic, target_critic, tau)

        # Update episode reward
        total_episode_reward += reward

        # Update state for next iteration
        state = next_state
        t += 1

    # Append episode reward
    episode_reward_list.append(total_episode_reward)
    episode_number_of_steps.append(t)
    # Close environment
    env.close()

    # Updates the tqdm update bar with fresh information
    # (episode number, total reward of the last episode, total number of Steps
    # of the last episode, average reward, average number of steps)
    avg_reward = running_average(
        episode_reward_list, n_ep_running_average)[-1]
    EPISODES.set_description("Episode {} - Reward/Steps: {:.1f}/{} - Avg. Reward/Steps: {:.1f}/{}".format(
        i, total_episode_reward, t,
        avg_reward,
        running_average(
            episode_number_of_steps, n_ep_running_average)[-1]))

if save_models:
    T.save(critic, 'best_critic.pt')
    T.save(actor, 'best_actor.pt')


# Plot Rewards and steps
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 9))
ax[0].plot([i for i in range(1, N_episodes + 1)],
           episode_reward_list, label='Episode reward')
ax[0].plot([i for i in range(1, N_episodes + 1)], running_average(
    episode_reward_list, n_ep_running_average), label='Avg. episode reward')
ax[0].set_xlabel('Episodes')
ax[0].set_ylabel('Total reward')
ax[0].set_title('Total Reward vs Episodes')
ax[0].legend()
ax[0].grid(alpha=0.3)

ax[1].plot([i for i in range(1, N_episodes + 1)],
           episode_number_of_steps, label='Steps per episode')
ax[1].plot([i for i in range(1, N_episodes + 1)], running_average(
    episode_number_of_steps, n_ep_running_average), label='Avg. number of steps per episode')
ax[1].set_xlabel('Episodes')
ax[1].set_ylabel('Total number of steps')
ax[1].set_title('Total number of steps vs Episodes')
ax[1].legend()
ax[1].grid(alpha=0.3)
plt.savefig(str(discount_factor) + ',' + str(L) + ',' + str(N_episodes) +
            ',' + str(N) + ',' + '.png')
plt.show()
