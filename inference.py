import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from scipy.special import logsumexp
from task_generation import *


if __name__ == '__main__':
    transition_matrix, cz_list, cz_to_ind_hash = get_c_z_transition_matrix(task_dict, p_stay=0.8)
    x_set = [0, 1]

    tasks = list(task_dict.keys())
    sigma = 0.1
    gdth_task = np.random.choice(tasks)
    gdth_x = np.random.choice(x_set)
    sy, boundaries = compose_trial(task_dict[gdth_task], {'theta_task': gdth_x}, sigma, sigma)
    s_arr, y_arr = sy['s'], sy['y']

    ##################### given s, y, cue, infer c, z, x  ####################
    ##################### log likelihood and p0  ####################
    nT = len(s_arr)
    log_likes = np.zeros((nT, len(x_set), len(cz_list)))

    for ix, x in enumerate(x_set):
        for icz, (c, z) in enumerate(cz_list):
            log_likes[:, ix, icz] = compute_emission_logp(s_arr, y_arr, z, {'theta_task': x},
                                                          sigma_s=sigma, sigma_y=sigma)

    log_eps = 1e-10
    p0_cz = np.zeros(len(cz_list))
    for task in tasks:
        p0_cz[cz_list.index((task, 'F'))] = 1 / len(tasks)
    p0_cz += log_eps
    p0_cz /= p0_cz.sum()

    ####################### viterbi ##############################
    scores = np.zeros((nT, len(x_set), len(cz_list)))
    args = np.zeros((nT, len(x_set), len(cz_list)), dtype=int)
    for it in range(nT - 2, -1, -1):
        vals = (scores[it + 1] + log_likes[it + 1]).T
        vals = np.log(transition_matrix[:, :, None] + log_eps) + vals[None, :, :]
        for ix in range(len(x_set)):
            for icz in range(len(cz_list)):
                args[it + 1, ix, icz] = np.argmax(vals[icz, :, ix])
                scores[it, ix, icz] = np.max(vals[icz, :, ix])

    scores_ = np.log(p0_cz) + scores[0] + log_likes[0]
    ind_ = np.unravel_index(np.argmax(scores_, axis=None), scores_.shape)
    x_ = x_set[ind_[0]]
    cz_ = np.zeros(nT, dtype=int)
    cz_[0] = ind_[1]
    for it in range(1, nT):
        cz_[it] = args[it, ind_[0], cz_[it - 1]]
    ibds = [0] + list(np.argwhere(np.diff(cz_)).flatten() + 1)
    print(ibds)
    print([cz_list[cz_[it]] for it in ibds])
    print(x_)
    print(boundaries)
    print(gdth_task, task_dict[gdth_task])
    print(gdth_x)

    ################### forward backward ##############################
    # alphas = np.zeros(log_likes.shape)
    # alphas[0] = np.log(p0_cz) + log_likes[0]
    # for it in range(nT - 1):
    #     m = np.max(alphas[it])
    #     with np.errstate(divide='ignore'):
    #         alphas[it + 1] = np.log(np.dot(np.exp(alphas[it] - m), transition_mtrx)) + m + log_likes[it + 1]
    # normalizer = logsumexp(alphas[-1])
    # betas = np.zeros(log_likes.shape)
    # betas[nT - 1] = 0
    # for it in range(nT - 2, -1, -1):
    #     tmp = (log_likes[it + 1] + betas[it + 1]).T
    #     m = np.max(tmp)
    #     betas[it] = (np.log(np.dot(transition_mtrx, np.exp(tmp - m))) + m).T
    # expected_states = alphas + betas
    # expected_states -= logsumexp(expected_states, axis=(-1, -2), keepdims=True)
    # expected_states = np.exp(expected_states)