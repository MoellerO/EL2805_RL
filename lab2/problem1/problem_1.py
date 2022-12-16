# Authors: Jing Li (19980512-9260) and Oliver Möller (19980831-2913)

# Load packages
from DQN_agent import *
from tqdm import trange
import matplotlib as mpl
import matplotlib.pyplot as plt

# Parameters
n_episodes = 50
discount_factor = 0.98
n_ep_running_average = 50

learning_rate = 0.0005
batch_size = 50
buffer_size = 10000

min_epsilon = 0.05
max_epsilon = 0.99
Z = 0.925


def epsilon_decay_exponential(k, N_episodes):
    epsilon = max(min_epsilon, max_epsilon * (min_epsilon / max_epsilon) ** ((k - 1) / (N_episodes * Z - 1)))
    return epsilon


def running_average(x, N):
    ''' Function used to compute the running average
        of the last N elements of a vector x
    '''
    if len(x) >= N:
        y = np.copy(x)
        y[N-1:] = np.convolve(x, np.ones((N, )) / N, mode='valid')
    else:
        y = np.zeros_like(x)
    return y
    

# Given a pth file and behave according to the network
def test_agent(env, agent, N):
    env.reset()
    episode_reward_list = []
    episode_number_of_steps = []

    EPISODES = trange(N, desc='Episodes', leave=True)
    for i in EPISODES:
        done = False
        state = env.reset()
        total_episode_reward = 0
        t = 0

        while not done:
            action = agent.forward(state)
            # Get next state, reward and done. Append into a buffer
            next_state, reward, done, _ = env.step(action)
            # Update episode reward
            total_episode_reward += reward
            # Update state for next iteration
            state = next_state
            t += 1

        # Append episode reward and total number of steps
        episode_reward_list.append(total_episode_reward)
        episode_number_of_steps.append(t)

        # Close environment
        env.close()

    return N, episode_reward_list, episode_number_of_steps


def random_vs_agent(n_episodes, filename = 'neural-network-1.pth'):
    env = gym.make('LunarLander-v2')

    dqn_agent = Agent(filename)
    num_episodes_dqn, dqn_rewards, _ = test_agent(env, dqn_agent, n_episodes)
    
    random_agent = RandomAgent(env.action_space.n)
    num_episodes_random, random_rewards, _ = test_agent(env, random_agent, n_episodes)

    fig = plt.figure(figsize=(10, 10))

    x_random = [i for i in range(1, num_episodes_random+1)]
    x_dqn = [i for i in range(1, num_episodes_dqn+1)]
    plt.plot(x_random, random_rewards, label='Random Agent', linestyle='dashed', color='red')
    plt.plot(x_dqn, dqn_rewards, label='DQN Agent', color='blue')
    plt.ylim(-400, 400)
    plt.xlabel('Episodes')
    plt.ylabel('Rewards')
    plt.title('Rewards vs Episodes')
    plt.legend()
    plt.savefig('random_comparison.png')


def train(N_episodes, discount_factor, n_ep_running_average, learning_rate, batch_size, buffer_size, filename):
    env = gym.make('LunarLander-v2')
    env.reset()

    # Number of actions
    n_actions = env.action_space.n
    # State dimensionality
    dim_state = len(env.observation_space.high)
    # If C steps have passed, set the target network equal to the main network
    C = buffer_size/batch_size

    # Total reward per episode
    episode_reward_list = []
    # Number of steps per episode
    episode_number_of_steps = []

    # Used for showing training progress
    EPISODES = trange(N_episodes, desc='Episode: ', leave=True)

    buffer = ExperienceReplayBuffer(maximum_length=buffer_size)

    agent = DQNAgent(batch_size, discount_factor, learning_rate, n_actions, dim_state)

    for i in EPISODES:
        # Initialize state
        state = env.reset()
        # Initialize done
        done = False
        # Initialize episodic reward
        total_episode_reward = 0.
        # Initialize step
        step = 0
        count = 1
        epsilon = epsilon_decay_exponential(i + 1, N_episodes)

        while not done:
            action = agent.forward(state, epsilon)
            # print('changed')
            next_state, reward, done, _ = env.step(action)
            buffer.append((state, action, reward, next_state, done))

            if len(buffer) > batch_size:
                agent.train(buffer)

            if count % C == 0:
                agent.update_ann()
                # print(f"Frame index is {count}")

            total_episode_reward += reward
            state = next_state
            step += 1
            count += 1

        episode_reward_list.append(total_episode_reward)
        episode_number_of_steps.append(step)

        env.close()

        EPISODES.set_description(
            "Episode {} - Reward/Steps: {:.1f}/{} - Avg. Reward/Steps: {:.1f}/{}".format(
            i, total_episode_reward, step,
            running_average(episode_reward_list, n_ep_running_average)[-1],
            running_average(episode_number_of_steps, n_ep_running_average)[-1]))

    draw_plots(N_episodes, episode_reward_list, n_ep_running_average, episode_number_of_steps)
    # agent.save_ann(filename_main=filename)


def draw_plots(N_episodes, episode_reward_list, n_ep_running_average, episode_number_of_steps):
    """ Draw total reward and number of steps plot
    """
    # Plot Rewards and steps
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 9))
    ax[0].plot([i for i in range(1, N_episodes+1)], episode_reward_list, label='Episodic reward')
    ax[0].plot([i for i in range(1, N_episodes+1)], running_average(
        episode_reward_list, n_ep_running_average), label='Average episodic reward')
    ax[0].set_xlabel('Episodes')
    ax[0].set_ylabel('Total episodic reward')
    ax[0].set_title('Total Episodic Reward vs Episodes')
    ax[0].legend()
    ax[0].grid(alpha=0.3)

    ax[1].plot([i for i in range(1, N_episodes+1)], episode_number_of_steps, label='Steps per episode')
    ax[1].plot([i for i in range(1, N_episodes+1)], running_average(
        episode_number_of_steps, n_ep_running_average), label='Average number of steps per episode')
    ax[1].set_xlabel('Episodes')
    ax[1].set_ylabel('Total number of steps')
    ax[1].set_title('Total number of steps vs Episodes')
    ax[1].legend()
    ax[1].grid(alpha=0.3)
    plt.savefig('e_3_memory.png')
    # plt.show()


def plot_policy_3D(network):
    y_values = np.linspace(0, 1.5, 100)
    w_values = np.linspace(-np.pi, np.pi, 100)

    Y, W = np.meshgrid(y_values, w_values)

    network = torch.load(network)
    
    # Initialize the Q matrix and action matrix
    dim = (len(y_values), len(w_values))
    Q = np.zeros(dim)
    action = np.zeros(dim)
    
    for y_idx, y in enumerate(y_values):
        for w_idx, w in enumerate(w_values):
            state = torch.tensor((0, y, 0, 0, w, 0, 0, 0), dtype=torch.float32)
            Q[w_idx, y_idx] = network(state).max(0)[0].item()
            action[w_idx,y_idx] = torch.argmax(network(state)).item()

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(W, Y, Q, cmap=mpl.cm.coolwarm)
    ax.set_ylabel('Height y')
    ax.set_xlabel('Angle ω')
    ax.set_zlabel('Value of optimal policy V')
    plt.savefig('3D_optimal_policy.png')
    # plt.show()

    fig2 = plt.figure()
    ax2 = fig2.gca(projection='3d')
    surf2 = ax2.plot_surface(W, Y, action)
    ax2.set_ylabel('Height y')
    ax2.set_xlabel('Angle ω')
    ax2.set_zlabel('Best Action')
    plt.savefig('3D_best_action_1.png')
    # plt.show()

    fig3 = plt.figure()
    ax3 = fig3.add_subplot(111)
    im = ax3.pcolormesh(Y, W, action)
    cbar = fig3.colorbar(im, ticks=[0, 1, 2, 3])
    ax3.set_ylabel('Height y')
    ax3.set_xlabel('Angle ω')
    ax3.set_title('Best Action')
    plt.savefig('3D_best_action_2.png')
    # plt.show()


#### Training #####
train_ex = True
print(';) Discount factor is: ', discount_factor)

filename = 'neural-network-1.pth'
# train(n_episodes, discount_factor, n_ep_running_average, learning_rate, batch_size, buffer_size, filename)
# plot_policy_3D(filename)
random_vs_agent(n_episodes, filename = 'neural-network-1.pth')
