import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from scipy.special import logsumexp
from task_generation import *
from inference import forward_backward

# task_dict = {
#     'PRO_D': 'F/D->S->R_P', # "DelayPro": respond to the stimulus once fixation is off
#     'PRO_M': 'F/D->S->F/D->R_M_P', # "MemoryPro" respond to the stimulus after a memory period
#     'PRO_R': 'F/D->R_P', # "ResponsePro" respond to the stimulus ASAP
#     'ANTI_D': 'F/D->S->R_A', # "DelayAnti": respond to the opposite once fixation is off
#     'ANTI_M': 'F/D->S->F/D->R_M_A', # "MemoryAnti": respond to the opposite after memory period
#     'ANTI_R': 'F/D->R_A', # "ResponseAnti": respond to the opposite ASAP
#     'ORDER': 'F/D->S->S_B->R_M_P', # respond to the stimulus coming on first
#     'DM': 'F/D->S_S->R_M_P' # respond to the stronger stimulus
# }
# for x in range(2):
#     w_table[x, 0, :] = [0, 0, 1, 0, 0, 1]
#     w_table[x, 1, :] = [1 - x, x, 1, 0, 0, 1]
#     w_table[x, 2, :] = [x, 1 - x, 1, 0, 0, 1]
#     w_table[x, 3, :] = [1, 1, 1, 0, 0, 1]
#     w_table[x, 4, :] = [1 - x + x * 0.5, x + (1 - x) * 0.5, 1, 0, 0, 1]
#     w_table[x, 5, :] = [1 - x, x, 0, 1 - x, x, 0]
#     w_table[x, 6, :] = [x, 1 - x, 0, 1 - x, x, 0]
#     w_table[x, 7, :] = [0, 0, 0, 1 - x, x, 0]

if __name__ == '__main__':
    np.random.seed(6)
    tasks = list(task_dict.keys())
    x_set = [0, 1]
    sigma = 0.01
    p_stay = 0.9
    min_Te = 5
    trials = []
    # for itrial in range(nc * 4):
    #     c = np.mod(itrial, nc)
    #     x = np.mod(itrial // nc, 2)
    gdth_c_list = [2]
    nc = len(gdth_c_list)
    for ibatch in range(1):
        for c, gdth_c in enumerate(gdth_c_list):
            for x in x_set:
                sy, boundaries = compose_trial(task_dict[tasks[gdth_c]], {'theta_task': x}, sigma, sigma,
                                               p_stay, min_Te)
                sy['s'] = sy['s'][:, :2]
                sy['y'] = sy['y'][:, :2]
                trials.append((c, x, sy))
    ####################################
    obsv_dim = 4
    z_list = ['F/D', 'R_P']#, 'S', 'R_M_P']#, 'S_B']#, 'S_S']
    nx = len(x_set)
    nz = len(z_list)
    eps = 1e-10
    p0_z = np.ones(nz) / nz
    ###################################
    w_table = np.random.rand(nx, nz, obsv_dim)
    # for x in x_set:
    #     w_table[x, 0, :] = [0, 0, 0, 0] #[0, 0, 1, 0, 0, 1]
    #     w_table[x, 1, :] = [1 - x, x, 1 - x, x]  # [1 - x, x, 0, 1 - x, x, 0]
        # w_table[x, 2, :] = [1 - x, x, 0, 0]
    ###################################
    p_stay = 0.9
    transition_matrix = np.ones((nc, nz, nz)) * (1 - p_stay) / (nz - 1)
    transition_matrix[:, np.arange(nz), np.arange(nz)] = p_stay
    # transition_matrix = np.random.rand(nc, nz, nz)
    # transition_matrix /= transition_matrix.sum(-1, keepdims=True)
    logp_obsv_arr = []
    tol = 0.0
    for irun in range(15):
        cur_logp_obsv = 0
        p0_z_ = np.zeros(nz)
        tmtrx = np.zeros((nc, nz, nz))
        w_table_numer = np.zeros((nx, nz, obsv_dim))
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
            # gamma, xi, p_cx, logp_obsv = forward_backward(p0_z, transition_matrix[c:(c+1)], log_likes)
            # prior_cx = np.ones((nc, nx)) / (nc * nx)
            prior_cx = np.zeros((nc, nx)) + eps
            prior_cx[c] = 1
            prior_cx /= prior_cx.sum()
            gamma, xi, p_cx, logp_obsv = forward_backward(p0_z, transition_matrix, log_likes, prior_cx=prior_cx)
            # nc, nx, nz, nT; # nc, nx, nz, nz, nT-1; # nc, nx;
            cur_logp_obsv += logp_obsv
            p0_z_ += gamma[..., 0].sum((0, 1))
            tmtrx += xi.sum((1, -1))
            w_table_numer += gamma.sum(0) @ w_arr  # nx, nz, d
            w_table_denom += gamma.sum((0, -1))
        ################# update ################
        logp_obsv_arr.append(cur_logp_obsv)
        if len(logp_obsv_arr) > 2:
            if np.abs(logp_obsv_arr[-1] - logp_obsv_arr[-2]) < tol:
                break
        p0_z_ += eps
        p0_z = p0_z_ / p0_z_.sum()
        tmtrx += eps
        transition_matrix = tmtrx / tmtrx.sum(axis=-1, keepdims=True)
        w_table = w_table_numer / (w_table_denom[..., None] + eps)

    fig, ax = plt.subplots(1, 1, figsize=(2, 2))
    ax.plot(logp_obsv_arr)
    fig.show()

    fig, axes = plt.subplots(1, nc, figsize=(2 * nc, 2))
    if nc == 1:
        axes = [axes]
    axes[0].set_ylabel('z')
    for i in range(nc):
        axes[i].imshow(np.log(transition_matrix[i]))
        axes[i].set_title(f'task {i}')
        axes[0].set_xlabel('z')
    fig.show()

    fig, axes = plt.subplots(1, len(x_set), figsize=(8, 4))
    if len(x_set) == 1:
        axes = [axes]
    for ix in range(len(x_set)):
        axes[ix].imshow(w_table[x_set[ix]], vmin=0, vmax=1)
        axes[ix].set_title(f'x={x_set[ix]}')
        axes[ix].set_xlabel('(s, y)')
        axes[ix].set_ylabel('z')
    fig.show()
    print(logp_obsv_arr[-1])
    print(np.argwhere(np.diff(logp_obsv_arr) < -eps).flatten())

    # for trial in trials:
    #     fig, axes = plt.subplots(2, 1, figsize=(5, 2))
    #     for i in range(2):
    #         axes[i].plot(trial[-1]['s'][:, i])
    #         axes[i].plot(trial[-1]['y'][:, i])
    #     fig.show()
