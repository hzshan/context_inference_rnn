import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from scipy.special import logsumexp
from task_generation import *
from inference import forward_backward


if __name__ == '__main__':
    tasks = list(task_dict.keys())
    x_set = [0, 1]
    sigma = 0.1
    trials = []
    for itrial in range(32):
        c = np.mod(itrial, 8)
        x = np.mod(itrial // 8, 2)
        sy, boundaries = compose_trial(task_dict[tasks[c]], {'theta_task': x}, sigma, sigma)
        trials.append((c, x, sy))
    ####################################
    z_list = ['F', 'S_P', 'S_A', 'S_B', 'S_S', 'R_P', 'R_A', 'R_M']
    nx = 2
    nc = 8
    nz = len(z_list)
    eps = 1e-10
    p0_z = np.zeros(nz)
    p0_z[0] = 1
    p0_z += eps
    p0_z /= p0_z.sum()
    ###################################
    w_table = np.random.rand(nx, nz, 6)
    # for x in range(2):
    #     w_table[x, 0, :] = [0, 0, 1, 0, 0, 1]
    #     w_table[x, 1, :] = [1 - x, x, 1, 0, 0, 1]
    #     w_table[x, 2, :] = [x, 1 - x, 1, 0, 0, 1]
    #     w_table[x, 3, :] = [1, 1, 1, 0, 0, 1]
    #     w_table[x, 4, :] = [1 - x + x * 0.5, x + (1 - x) * 0.5, 1, 0, 0, 1]
    #     w_table[x, 5, :] = [1 - x, x, 0, 1 - x, x, 0]
    #     w_table[x, 6, :] = [x, 1 - x, 0, 1 - x, x, 0]
    #     w_table[x, 7, :] = [0, 0, 0, 1 - x, x, 0]
    for x in range(2):
        w_table[x, 0, :] = [0, 0, 1, 0, 0, 1]
        for iz in range(5):
            w_table[x, iz, -4:] = [1, 0, 0, 1]
        for iz in range(5, 8):
            w_table[x, iz, -4:] = [0, 1 - x, x, 0]
    ###################################
    # p_stay = 0.9
    # transition_matrix = np.ones((nc, nz, nz)) * (1 - p_stay) / (nz - 1)
    # transition_matrix[:, np.arange(nz), np.arange(nz)] = p_stay
    transition_matrix = np.random.rand(nc, nz, nz)
    transition_matrix /= transition_matrix.sum(-1, keepdims=True)
    logp_obsv_arr = []
    tol = 0.001
    for irun in range(100):
        cur_logp_obsv = 0
        tmtrx = np.zeros((nc, nz, nz))
        w_table_numer = np.zeros((nx, nz, 6))
        w_table_denom = np.zeros((nx, nz))
        for itrial in range(len(trials)):
            c, gdth_x, sy = trials[itrial]
            w_arr = np.concatenate((sy['s'], sy['y']), axis=-1)
            nT = len(w_arr)
            log_likes = np.zeros((nx, nz, nT))
            for ix, x in enumerate(x_set):
                for iz in range(nz):
                    log_likes[ix, iz, :] = multivariate_log_likelihood(obs=w_arr, mean=w_table[ix, iz][None, :],
                                                                       sigma=sigma)
            gamma, xi, p_cx, logp_obsv = forward_backward(p0_z, transition_matrix, log_likes)
            # nc, nx, nz, nT; # nc, nx, nz, nz, nT-1; # nc, nx;
            cur_logp_obsv += logp_obsv
            tmtrx[c] += xi.sum((1, -1))[c] / (p_cx.sum(-1)[c] + eps)
            w_table_numer += gamma.sum(0) @ w_arr  # nx, nz, d
            w_table_denom += gamma.sum((0, -1))
        ################# update ################
        logp_obsv_arr.append(cur_logp_obsv)
        if len(logp_obsv_arr) > 2:
            if np.abs(logp_obsv_arr[-1] - logp_obsv_arr[-2]) < tol:
                break
        tmtrx += eps
        transition_matrix = tmtrx / tmtrx.sum(axis=-1, keepdims=True)
        w_table = w_table_numer / (w_table_denom[..., None] + eps)
        for x in range(2):
            w_table[x, 0, :] = [0, 0, 1, 0, 0, 1]
            for iz in range(5):
                w_table[x, iz, -4:] = [1, 0, 0, 1]
            for iz in range(5, 8):
                w_table[x, iz, -4:] = [0, 1 - x, x, 0]

    fig, ax = plt.subplots(1, 1, figsize=(2, 2))
    ax.plot(logp_obsv_arr)
    fig.show()

    fig, axes = plt.subplots(1, nc, figsize=(2 * nc, 2))
    axes[0].set_ylabel('z')
    for i in range(nc):
        axes[i].imshow(np.log(transition_matrix[i]))
        axes[i].set_title(f'task {i}')
        axes[0].set_xlabel('z')
    fig.show()

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    for ix in range(2):
        axes[ix].imshow(w_table[x_set[ix]], vmin=0, vmax=1)
        axes[ix].set_title(f'x={x_set[ix]}')
        axes[ix].set_xlabel('(s, y)')
        axes[ix].set_ylabel('z')
    fig.show()