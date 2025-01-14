import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from scipy.special import logsumexp
from task_generation import *


def viterbi(p0_z, transition_matrix, log_likes):
    nc, nz, _ = transition_matrix.shape
    nx, _, nT = log_likes.shape
    scores = np.zeros((nc, nx, nz, nT))
    args = np.zeros((nc, nx, nz, nT), dtype=int)
    for it in range(nT - 2, -1, -1):
        vals = scores[..., it + 1] + log_likes[..., it + 1]
        vals = np.log(transition_matrix[:, None, :, :]) + vals[:, :, None, :]
        args[..., it + 1] = np.argmax(vals, axis=-1)
        scores[..., it] = np.max(vals, axis=-1)
    scores_ = np.log(p0_z) + scores[..., 0] + log_likes[..., 0]
    ind_ = np.unravel_index(np.argmax(scores_, axis=None), scores_.shape)
    z_ = np.zeros(nT, dtype=int)
    z_[0] = ind_[2]
    for it in range(1, nT):
        z_[it] = args[ind_[0], ind_[1], z_[it - 1], it]
    return ind_, z_


def forward_backward(p0_z, transition_matrix, log_likes):
    nc, nz, _ = transition_matrix.shape
    nx, _, nT = log_likes.shape
    alphas = np.zeros((nc, nx, nz, nT))
    alphas[..., 0] = np.log(p0_z) + log_likes[..., 0]
    for it in range(nT - 1):
        m = np.max(alphas[..., it], axis=-1, keepdims=True)
        with np.errstate(divide='ignore'):
            alphas[..., it + 1] = np.log(np.einsum('cxi,cij->cxj', np.exp(alphas[..., it] - m),
                                                   transition_matrix)) + m + log_likes[..., it + 1]
    normalizer = logsumexp(alphas[..., -1], axis=-1)
    betas = np.zeros((nc, nx, nz, nT))
    betas[..., nT - 1] = 0
    for it in range(nT - 2, -1, -1):
        tmp = log_likes[..., it + 1] + betas[..., it + 1]
        m = np.max(tmp, axis=-1, keepdims=True)
        with np.errstate(divide='ignore'):
            betas[..., it] = np.log(np.einsum('cij,cxj->cxi', transition_matrix,
                                              np.exp(tmp - m))) + m
    expected_states = alphas + betas
    expected_states -= logsumexp(expected_states, axis=-2, keepdims=True)
    expected_states = np.exp(expected_states)
    return expected_states, normalizer


if __name__ == '__main__':
    tasks = list(task_dict.keys())
    x_set = [0, 1]
    sigma = 0.1
    gdth_task = np.random.choice(tasks)
    gdth_x = np.random.choice(x_set)
    sy, boundaries = compose_trial(task_dict[gdth_task], {'theta_task': gdth_x}, sigma, sigma)
    s_arr, y_arr = sy['s'], sy['y']
    print(gdth_task, task_dict[gdth_task].split('->'))
    print(gdth_x)
    print(boundaries[:-1])
    ##################### given s, y, cue, infer c, z, x  ####################
    ##################### p0, transition matrix and log likelihood  ####################
    log_eps = 1e-10
    transition_matrix, _, z_list = get_z_transition_matrix_per_task(task_dict, p_stay=0.99)
    transition_matrix += log_eps
    transition_matrix /= transition_matrix.sum(axis=-1, keepdims=True)
    p0_z = np.zeros(len(z_list))
    p0_z[z_list.index('F')] = 1
    p0_z += log_eps
    p0_z /= p0_z.sum()
    nT = len(s_arr)
    log_likes = np.zeros((len(x_set), len(z_list), nT))
    for ix, x in enumerate(x_set):
        for iz, z in enumerate(z_list):
            log_likes[ix, iz, :] = compute_emission_logp(s_arr, y_arr, z, {'theta_task': x},
                                                         sigma_s=sigma, sigma_y=sigma)
    ####################### viterbi ##############################
    ind_, z_ = viterbi(p0_z, transition_matrix, log_likes)
    c_ = tasks[ind_[0]]
    x_ = x_set[ind_[1]]
    ibds = [0] + list(np.argwhere(np.diff(z_)).flatten() + 1)
    print(c_, [z_list[z_[it]] for it in ibds])
    print(x_)
    print(ibds[1:])
    # ################### forward backward ##############################
    expected_states, normalizer = forward_backward(p0_z, transition_matrix, log_likes)
    ind_ = np.unravel_index(np.argmax(normalizer), normalizer.shape)
    c_ = tasks[ind_[0]]
    x_ = x_set[ind_[1]]
    z_ = np.argmax(expected_states[ind_[0], ind_[1], :, :], axis=0)
    ibds = [0] + list(np.argwhere(np.diff(z_)).flatten() + 1)
    print(c_, [z_list[z_[it]] for it in ibds])
    print(x_)
    print(ibds[1:])