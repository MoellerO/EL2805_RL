import numpy as np
import maze as mz

field = np.array([
    [0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 1, 0, 0],
    [0, 0, 1, 0, 0, 1, 1, 1],
    [0, 0, 1, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 1, 1, 1, 1, 0],
    [0, 0, 0, 0, 1, 2, 0, 0]
])

# mz.draw_maze(maze)
env = mz.Maze(field)
# print(env.n_states)
# print(env.rewards)
# print(env.minotaur)

# Finite horizon
horizon = 20
# Solve the MDP problem with dynamic programming
V, policy = mz.dynamic_programming(env, horizon)
method = 'DynProg'
# first two digits are player positions, last two minotaur
start = (0, 0, 6, 6)
path = env.simulate(start, policy, method)
print(path)
# Show the shortest path
mz.animate_solution(field, path)
