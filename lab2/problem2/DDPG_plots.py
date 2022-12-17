# Authors: Jing Li (19980512-9260) and Oliver Möller (19980831-2913)

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch

dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def plot_policy_3D(actor):
    y_values = np.linspace(0, 1.5, 100)
    w_values = np.linspace(-np.pi, np.pi, 100)

    Y, W = np.meshgrid(y_values, w_values)

    # network = torch.load(network)
    actor = torch.load('neural-network-2-actor.pt',
                       map_location=torch.device('cpu'))
    critic = torch.load('neural-network-2-critic.pt',
                        map_location=torch.device('cpu'))

    # Initialize the Q matrix and action matrix
    dim = (len(y_values), len(w_values))
    Q = np.zeros(dim)
    action = np.zeros(dim)

    for y_idx, y in enumerate(y_values):
        for w_idx, w in enumerate(w_values):

            state = torch.tensor([(0, y, 0, 0, w, 0, 0, 0)],
                                 dtype=torch.float32).to(dev)

            # act_state_batch = torch.tensor(
            #     [state], requires_grad=False, dtype=torch.float32).to(dev)
            # action = actor.forward(act_state_batch)

            pi = actor(state)
            q = critic(state, pi)
            Q[w_idx, y_idx] = q.item()
            action[w_idx, y_idx] = pi[0][1].item()

    fig = plt.figure()
    # ax = fig.gca(projection='3d')
    ax = fig.add_subplot(projection='3d')
    surf = ax.plot_surface(W, Y, Q, cmap=mpl.cm.coolwarm)
    ax.set_ylabel('Height y')
    ax.set_xlabel('Angle ω')
    ax.set_zlabel('Value of optimal policy V')
    plt.savefig('3D_optimal_policy.png')
    plt.show()

    fig2 = plt.figure()
    # ax2 = fig2.gca(projection='3d')
    ax2 = fig2.add_subplot(projection='3d')
    surf2 = ax2.plot_surface(W, Y, action)
    ax2.set_ylabel('Height y')
    ax2.set_xlabel('Angle ω')
    ax2.set_zlabel('Best Action')
    plt.savefig('3D_best_action_1.png')
    plt.show()

    fig3 = plt.figure()
    ax3 = fig3.add_subplot(111)
    im = ax3.pcolormesh(Y, W, action)
    cbar = fig3.colorbar(im, ticks=[0, 1, 2, 3])
    ax3.set_ylabel('Height y')
    ax3.set_xlabel('Angle ω')
    ax3.set_title('Best Action')
    plt.savefig('3D_best_action_2.png')
    plt.show()


filename = 'neural-network-1.pth'
plot_policy_3D(filename)
