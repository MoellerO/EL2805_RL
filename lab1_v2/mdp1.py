import numpy as np
import matplotlib.pyplot as plt
import time
from IPython import display
import random

# Implemented methods
methods = ['DynProg', 'ValIter']

# Some colours
LIGHT_RED = '#FFC4CC'
LIGHT_GREEN = '#95FD99'
BLACK = '#000000'
WHITE = '#FFFFFF'
LIGHT_PURPLE = '#E8D0FF'
LIGHT_ORANGE = '#FAE0C3'


class Maze:
    # Actions
    STAY = 0
    MOVE_LEFT = 1
    MOVE_RIGHT = 2
    MOVE_UP = 3
    MOVE_DOWN = 4

    # Give names to actions
    actions_names = {
        STAY: "stay",
        MOVE_LEFT: "move left",
        MOVE_RIGHT: "move right",
        MOVE_UP: "move up",
        MOVE_DOWN: "move down"
    }

    # Reward values
    STEP_REWARD = -1
    GOAL_REWARD = 0
    IMPOSSIBLE_REWARD = -100
    CLOSE_REWARD = -2  # Getting close to the minotaur

    def __init__(self, maze, minotaur_stay=False):
        """ Constructor of the environment Maze.
        """
        self.maze = maze
        self.minotaur_stay = minotaur_stay
        self.actions = self.__actions()
        self.states, self.map = self.__states()
        self.n_actions = len(self.actions)
        self.n_states = len(self.states)
        self.transition_probabilities = self.__transitions()
        self.rewards = self.__rewards()

    def __actions(self):
        actions = dict()
        actions[self.STAY] = (0, 0)
        actions[self.MOVE_LEFT] = (0, -1)
        actions[self.MOVE_RIGHT] = (0, 1)
        actions[self.MOVE_UP] = (-1, 0)
        actions[self.MOVE_DOWN] = (1, 0)
        return actions

    def __states(self):
        states = dict()
        map = dict()
        end = False
        s = 0
        for i in range(self.maze.shape[0]):
            for j in range(self.maze.shape[1]):
                for k in range(self.maze.shape[0]):
                    for l in range(self.maze.shape[1]):
                        # the player can not stand somewhere that is a wall
                        if self.maze[i, j] != 1:
                            states[s] = (i, j, k, l)
                            map[(i, j, k, l)] = s
                            s += 1
        return states, map

    def __move(self, state, action):
        """ Makes a step in the maze, given a current position and an action.
            If the action STAY or an inadmissible action is used, the agent stays in place.

            :return tuple next_cell: Position (x,y) on the maze that agent transitions to.
        """
        # Possible next state: store the ID of the states
        next_state_list = []

        # If the player meets the minotaur, then no change in state
        if self.states[state][0:2] == self.states[state][2:]:
            count = 1
            next_state_list.append(state)
        else:
            # Compute the future position of the player given current (state, action)
            row = self.states[state][0] + self.actions[action][0]
            col = self.states[state][1] + self.actions[action][1]
            # Is the future position an impossible one ?
            hitting_maze_walls = (row == -1) or (row == self.maze.shape[0]) or \
                                 (col == -1) or (col == self.maze.shape[1]) or \
                                 (self.maze[row, col] == 1)

            # If hit the wall, stay still, else move according to the action
            if hitting_maze_walls:
                player_pos = self.states[state][0:2]
            else:
                player_pos = (row, col)

            if self.minotaur_stay:
                start_idx = 0
            else:
                start_idx = 1

            # Set a counter to count legal moving directions of the minotaur
            count = 0
            # Compute the future position of the minotaur given current (state, action)
            for move in range(start_idx, self.n_actions):
                row = self.states[state][2] + self.actions[move][0]
                col = self.states[state][3] + self.actions[move][1]
                hitting_maze_walls = (row == -1) or (row == self.maze.shape[0]) or \
                                     (col == -1) or (col == self.maze.shape[1])
                if not hitting_maze_walls:
                    next_state_list.append(
                        self.map[(player_pos[0], player_pos[1], row, col)])
                    count += 1

        p_minotaur = 1 / count

        return p_minotaur, next_state_list

    def __transitions(self):
        """ Computes the transition probabilities for every state action pair.
            :return numpy.tensor transition probabilities: tensor of transition
            probabilities of dimension S*S*A
        """
        # Initialize the transition probabilities tensor (S,S,A)
        dimensions = (self.n_states, self.n_states, self.n_actions)
        transition_probabilities = np.zeros(dimensions)

        # Compute the transition probabilities. Note that the transitions
        # are deterministic.
        for s in range(self.n_states):
            for a in range(self.n_actions):
                p_minotaur, next_state_list = self.__move(s, a)
                for next_s in next_state_list:
                    transition_probabilities[next_s, s, a] = 1 * p_minotaur
        return transition_probabilities

    def __rewards(self):
        """
        :return: the reward matrix
        """
        rewards = np.zeros((self.n_states, self.n_actions))

        for s in range(self.n_states):
            for a in range(self.n_actions):
                _, next_s_list = self.__move(s, a)
                reward_list = []
                for next_s in next_s_list:
                    # Reward for hitting a wall
                    if self.states[s][0:2] == self.states[next_s][0:2] and a != self.STAY:
                        reward_list.append(self.IMPOSSIBLE_REWARD)
                    # Reward for meeting the minotaur
                    elif self.states[next_s][0:2] == self.states[next_s][2:]:
                        reward_list.append(self.IMPOSSIBLE_REWARD)
                    # Reward for being next the Minotaur
                    elif (abs(self.states[next_s][0] - self.states[next_s][2]) +
                          abs(self.states[next_s][1] - self.states[next_s][3])) == 1:
                        reward_list.append(self.CLOSE_REWARD)
                    # Reward for reaching the exit
                    elif self.states[s][0:2] == self.states[next_s][0:2] and self.maze[self.states[next_s][0:2]] == 2:
                        reward_list.append(self.GOAL_REWARD)

                    # Reward for taking a step to an empty cell that is not the exit
                    else:
                        reward_list.append(self.STEP_REWARD)
                mean_reward = np.mean(reward_list)
                rewards[s, a] = mean_reward
        return rewards

    def simulate(self, start, policy, method):
        if method not in methods:
            error = 'ERROR: the argument method must be in {}'.format(methods)
            raise NameError(error)

        path = list()
        if method == 'DynProg':
            # Deduce the horizon from the policy shape
            horizon = policy.shape[1]

            # Initialize current state and time
            t = 0
            s = self.map[start]

            # Add the starting position in the maze to the path
            path.append(start)

            while t < horizon - 1:
                # Move to next state given the policy and the current state
                _, next_state_list = self.__move(s, policy[s, t])
                next_s = random.choice(next_state_list)

                # Add the position in the maze corresponding to the next state
                # to the path
                path.append(self.states[next_s])

                # Update time and state for next iteration
                t += 1
                s = next_s

                if self.states[s][0:2] == self.states[s][2:] or t == horizon - 1:
                    print("YOU LOST THE GAME")
                    break
                elif self.maze[self.states[s][0:2]] == 2:
                    print(f"YOU WON!!!! Current time step = {t}")
                    break
        ############### TO BE MODIFIED FOR e) ##############
        if method == 'ValIter':
            # Initialize current state, next state and time
            t = 1
            s = self.map[start]
            # Add the starting position in the maze to the path
            path.append(start)
            # Move to next state given the policy and the current state
            next_s = self.__move(s, policy[s])
            # Add the position in the maze corresponding to the next state
            # to the path
            path.append(self.states[next_s])
            # Loop while state is not the goal state
            while s != next_s:
                # Update state
                s = next_s
                # Move to next state given the policy and the current state
                next_s = self.__move(s, policy[s])
                # Add the position in the maze corresponding to the next state
                # to the path
                path.append(self.states[next_s])
                # Update time and state for next iteration
                t += 1
        return path

    def draw_path(self, policy_matrix, time_step, minotaur_pos):
        # print(len(self.states))  # 2240 states
        n_rows = self.maze.shape[0]
        n_cols = self.maze.shape[1]
        actions = np.zeros((n_rows, n_cols))

        for i, action in enumerate(policy_matrix[:, time_step]):
            if self.states[i][2:] == minotaur_pos:
                actions[self.states[i][:2]] = action
        draw_maze(self.maze, minotaur_pos, actions)


def dynamic_programming(env, horizon):
    """ Solves the shortest path problem using dynamic programming
        :input Maze env           : The maze environment in which we seek to
                                    find the shortest path.
        :input int horizon        : The time T up to which we solve the problem.
        :return numpy.array V     : Optimal values for every state at every
                                    time, dimension S*T
        :return numpy.array policy: Optimal time-varying policy at every state,
                                    dimension S*T
    """

    # The dynamic prgramming requires the knowledge of :
    # - Transition probabilities
    # - Rewards
    # - State space
    # - Action space
    # - The finite horizon
    p = env.transition_probabilities
    r = env.rewards
    n_states = env.n_states
    n_actions = env.n_actions
    T = horizon

    # The variables involved in the dynamic programming backwards recursions
    V = np.zeros((n_states, T + 1))
    policy = np.zeros((n_states, T + 1))
    Q = np.zeros((n_states, n_actions))

    # Initialization
    Q = np.copy(r)
    V[:, T] = np.max(Q, 1)
    policy[:, T] = np.argmax(Q, 1)

    # The dynamic programming backwards recursion
    for t in range(T - 1, -1, -1):
        # Update the value function according to the bellman equation
        for s in range(n_states):
            for a in range(n_actions):
                # Update of the temporary Q values
                Q[s, a] = r[s, a] + np.dot(p[:, s, a], V[:, t + 1])
        # Update by taking the maximum Q value w.r.t the action a
        V[:, t] = np.max(Q, 1)
        # The optimal action is the one that maximizes the Q function
        policy[:, t] = np.argmax(Q, 1)

    return V, policy


def draw_maze(maze, minotaur_pos, actions):
    # Map a color to each cell in the maze
    col_map = {0: WHITE, 1: BLACK,
               2: LIGHT_GREEN, -6: LIGHT_RED, -1: LIGHT_RED}

    # Give a color to each cell
    rows, cols = maze.shape
    colored_maze = [[col_map[maze[j, i]]
                     for i in range(cols)] for j in range(rows)]

    # Create figure of the size of the maze
    fig = plt.figure(1, figsize=(cols, rows))

    # Remove the axis ticks and add title
    ax = plt.gca()
    ax.set_title('The Maze')
    ax.set_xticks([])
    ax.set_yticks([])

    # Give a color to each cell
    rows, cols = maze.shape
    colored_maze = [[col_map[maze[j, i]]
                     for i in range(cols)] for j in range(rows)]

    # Create figure of the size of the maze
    fig = plt.figure(1, figsize=(cols, rows))

    # Create a table to color
    grid = plt.table(cellText=None,
                     cellColours=colored_maze,
                     cellLoc='center',
                     loc=(0, 0),
                     edges='closed')

    # Modify the height and width of the cells in the table
    tc = grid.properties()['children']
    for cell in tc:
        cell.set_height(1.0 / rows)
        cell.set_width(1.0 / cols)

    dict_actions = {0: 'Stay', 1: 'Left', 2: 'Right', 3: 'Up', 4: 'Down'}
    grid.get_celld()[minotaur_pos].set_facecolor(LIGHT_PURPLE)
    for i in range(actions.shape[0]):
        for j in range(actions.shape[1]):
            grid.get_celld()[(i, j)].get_text().set_text(
                dict_actions[actions[i, j]])


def animate_solution(maze, path):
    # Map a color to each cell in the maze
    col_map = {0: WHITE, 1: BLACK,
               2: LIGHT_GREEN, -6: LIGHT_RED, -1: LIGHT_RED}

    # Size of the maze
    rows, cols = maze.shape

    # Create figure of the size of the maze
    fig = plt.figure(1, figsize=(cols, rows))

    # Remove the axis ticks and add title
    ax = plt.gca()
    ax.set_title('Policy simulation')
    ax.set_xticks([])
    ax.set_yticks([])

    # Give a color to each cell
    colored_maze = [[col_map[maze[j, i]]
                     for i in range(cols)] for j in range(rows)]

    # Create figure of the size of the maze
    fig = plt.figure(1, figsize=(cols, rows))

    # Create a table to color
    grid = plt.table(cellText=None,
                     cellColours=colored_maze,
                     cellLoc='center',
                     loc=(0, 0),
                     edges='closed')

    # Modify the height and width of the cells in the table
    tc = grid.properties()['children']
    for cell in tc:
        cell.set_height(1.0 / rows)
        cell.set_width(1.0 / cols)

    # Update the color at each frame
    for i in range(len(path)):
        if i != 0:
            # If the player is at the exit
            if maze[path[i][0:2]] == 2:
                grid.get_celld()[(path[i][0:2])].set_facecolor(LIGHT_GREEN)
                grid.get_celld()[(path[i][0:2])].get_text().set_text(
                    'Player is out')
                grid.get_celld()[(path[i - 1][2:])
                                 ].set_facecolor(col_map[maze[path[i - 1][2:]]])
                grid.get_celld()[(path[i - 1][2:])].get_text().set_text('')
                grid.get_celld()[(path[i - 1][:2])
                                 ].set_facecolor(col_map[maze[path[i - 1][:2]]])
                grid.get_celld()[(path[i - 1][:2])].get_text().set_text('')
                break

            # If the player is not at the exit, change the previous cells (both the player
            # and the minotaur) back to their original colors
            else:
                grid.get_celld()[(path[i - 1][0:2])
                                 ].set_facecolor(col_map[maze[path[i - 1][0:2]]])
                grid.get_celld()[(path[i - 1][0:2])].get_text().set_text('')
                grid.get_celld()[(path[i - 1][2:])
                                 ].set_facecolor(col_map[maze[path[i - 1][2:]]])
                grid.get_celld()[(path[i - 1][2:])].get_text().set_text('')

        # Change the colors of the current cells
        grid.get_celld()[(path[i][0:2])].set_facecolor(LIGHT_ORANGE)
        grid.get_celld()[(path[i][0:2])].get_text().set_text('Player')

        grid.get_celld()[(path[i][2:])].set_facecolor(LIGHT_PURPLE)
        grid.get_celld()[(path[i][2:])].get_text().set_text('Minotaur')

        display.display(fig)
        display.clear_output(wait=True)
        time.sleep(1)
