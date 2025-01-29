import tqdm
import numpy as np
import inference

def offline_learning(init_transition_matrix,
                     init_w_table,
                     p0_z,
                     trials,
                     know_c=False,
                     max_iter=100,
                     tol=1e-5,
                     eps=1e-10,
                     sigma=0.01):

    nc, nz, _ = init_transition_matrix.shape
    nx = init_w_table.shape[0]

    w_table = init_w_table.copy()
    transition_matrix = init_transition_matrix.copy()

    logp_obsv_arr = []

    x_set = np.arange(nx)


    for irun in tqdm.trange(max_iter):
        cur_logp_obsv = 0
        tmtrx = np.zeros((nc, nz, nz))
        w_table_numer = np.zeros((nx, nz, 6))  # 6 is the dimension of stim + response
        w_table_denom = np.zeros((nx, nz))
        gammas = []

        # subset_trials = trials[np.random.choice(len(trials), 300)]

        for itrial in range(len(trials)):
            c, _, sy = trials[itrial]
            w_arr = np.concatenate((sy['s'], sy['y']), axis=-1)
            nT = len(w_arr)
            log_likes = np.zeros((nx, nz, nT))
            for ix, x in enumerate(x_set):
                for iz in range(nz):
                    log_likes[ix, iz, :] = \
                        inference.multivariate_log_likelihood(
                            obs=w_arr, mean=w_table[ix, iz][None, :],
                            sigma=sigma)

            gamma, xi, p_cx, logp_obsv = inference.forward_backward(
                p0_z, transition_matrix, log_likes)
            
            gammas.append(gamma)

            # nc, nx, nz, nT; # nc, nx, nz, nz, nT-1; # nc, nx;
            cur_logp_obsv += logp_obsv

            if know_c:
                tmtrx[c] += xi.sum((1, -1))[c] / (p_cx.sum(-1)[c] + eps)
            else:
                tmtrx += xi.sum((1, -1))
            w_table_numer += gamma.sum(0) @ w_arr  # nx, nz, d
            w_table_denom += gamma.sum((0, -1))

        ################# update ################

        # check convergence
        logp_obsv_arr.append(cur_logp_obsv)
        if len(logp_obsv_arr) > 2:
            if np.abs(logp_obsv_arr[-1] - logp_obsv_arr[-2]) < tol:
                print('converged')
                break
        tmtrx += eps
        transition_matrix = tmtrx / tmtrx.sum(axis=-1, keepdims=True)
        w_table = w_table_numer / (w_table_denom[..., None] + eps)

    return transition_matrix, w_table, logp_obsv_arr, gammas