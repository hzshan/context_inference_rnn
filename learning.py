import numpy as np
import inference

# Terminology: 
# LL: log likelihood of each trial
# total_LL: sum of LLs across trials
# LLpT: log likelihood per trial, total_LL / n_trials


def _update(orig, new, lr):
    return orig * (1 - lr) + new * lr
    # return orig + new

def offline_learning(init_M,
                     init_W,
                     init_p0_z,
                     trials,
                     max_iter=100,
                     tol=1e-5,
                     eps=1e-10,
                     sigma=0.01,
                     x_oracle=False):

    """
    Args:
        init_M: initial transition matrix p(z_t=i, z_{t+1}=j|c): shape (nc, nz, nz)
        init_W: initial weight table p(w_t|x, z_t=i): shape (nx, nz, d)
        init_p0_z: initial distribution of z p(z_1=i): shape (nz,)
        trials: list of trials with each trial being a tuple (c, x, sy)
        max_iter: maximum number of iterations
        tol: tolerance for convergence
        eps: small constant for numerical stability
        sigma: std of observation noise
    
    Returns:
        learned_M: learned transition matrix p(z_t=i, z_{t+1}=j|c): shape (nc, nz, nz)
        learned_W: learned weight table p(w_t|x, z_t=i): shape (nx, nz, d)
        logp_obsv_arr: log likelihood of all trials over time
        gamma_across_trials: list of gamma for each trial
    """

    nc, nz, _ = init_M.shape
    nx = init_W.shape[0]
    n_trials = len(trials)

    learned_W = init_W.copy()  # nx, nz, d
    learned_M = init_M.copy()  # nc, nz, nz
    learned_p0_z = init_p0_z.copy()  # nz

    LLpT_over_time = []

    for irun in range(max_iter):
        curr_total_LL = 0
        tmtrx = np.zeros((nc, nz, nz))
        w_table_numer = np.zeros((nx, nz, 6))  # 6 is the dimension of stim + response
        w_table_denom = np.zeros((nx, nz))
        gamma_across_trials = []

        for itrial in range(n_trials):
            _, _, sy = trials[itrial]
            w_arr = np.concatenate((sy['s'], sy['y']), axis=-1)

            gamma, xi, p_cx, LL = get_stats_for_single_trial(
                single_trial=trials[itrial],
                W=learned_W,
                M=learned_M,
                p0_z=learned_p0_z,
                x_oracle=x_oracle,  #TODO: add prior_cx
                sigma=sigma)
            
            gamma_across_trials.append(gamma)

            # nc, nx, nz, nT; # nc, nx, nz, nz, nT-1; # nc, nx;
            curr_total_LL += LL

            learned_p0_z += gamma[..., 0].sum((0, 1))
            tmtrx += xi.sum((1, -1))
            w_table_numer += gamma.sum(0) @ w_arr  # nx, nz, d
            w_table_denom += gamma.sum((0, -1))

        # check convergence
        LLpT_over_time.append(curr_total_LL / n_trials)
        if len(LLpT_over_time) > 2:
            if np.abs(LLpT_over_time[-1] - LLpT_over_time[-2]) < tol:
                break
        
        # update parameters
        learned_p0_z += eps
        learned_p0_z = learned_p0_z / learned_p0_z.sum()
        tmtrx += eps
        learned_M = tmtrx / tmtrx.sum(axis=-1, keepdims=True)
        learned_W = w_table_numer / (w_table_denom[..., None] + eps)
    
    # warn if not converged
    if irun == max_iter - 1:
        print('Warning: EM did not converge')

    return learned_M, learned_W, learned_p0_z, LLpT_over_time, gamma_across_trials


def online_learning(init_M,
                    init_W,
                    init_p0_z,
                    trials,
                    n_sweeps=1,
                    eps=1e-10,
                    sigma=0.01,
                    lr=0.1,
                    iters_per_trial=10,
                    x_oracle=False):
    """
    Args:
        init_M: initial transition matrix p(z_t=i, z_{t+1}=j|c): shape (nc, nz, nz)
        init_W: initial weight table p(w_t|x, z_t=i): shape (nx, nz, d)
        init_p0_z: initial distribution of z p(z_1=i): shape (nz,)
        trials: list of trials with each trial being a tuple (c, x, sy)
        n_sweeps: number of sweeps (for true online learning, set to 1)
        eps: small constant for numerical stability
        sigma: std of observation noise
        lr: learning rate
        iters_per_trial: number of iterations per trial
    
    Returns:
        learned_M: learned transition matrix p(z_t=i, z_{t+1}=j|c): shape (nc, nz, nz)
        learned_W: learned weight table p(w_t|x, z_t=i): shape (nx, nz, d)
        LLpT_over_time: log likelihood of all trials over time
        gamma_across_trials: list of gamma for each trial (using the final parameters)
    """

    nc, nz, _ = init_M.shape
    nx = init_W.shape[0]

    learned_W = init_W.copy()
    learned_M = init_M.copy()
    learned_p0_z = init_p0_z.copy()

    LLpT_over_time = []

    # initialize sufficient statistics
    tmtrx = np.zeros((nc, nz, nz))
    w_table_numer = np.zeros((nx, nz, 6))  # 6 is the dimension of stim + response
    w_table_denom = np.zeros((nx, nz))

    for i in range(n_sweeps):
        
        for itrial in range(len(trials)):
            
            # load trial
            c, x, sy = trials[itrial]
            w_arr = np.concatenate((sy['s'], sy['y']), axis=-1)

            # i guess we could initialize them at random as well
            learned_W_this_trial = learned_W.copy()
            learned_M_this_trial = learned_M.copy()
            learned_p0_z_this_trial = learned_p0_z.copy()

            w_table_numer_this_trial = w_table_numer.copy()
            w_table_denom_this_trial = w_table_denom.copy()
            tmtrx_this_trial = tmtrx.copy()

            for iiter in range(iters_per_trial):

                # get sufficient statistics for this trial
                gamma, xi, _, _ = get_stats_for_single_trial(
                    single_trial=trials[itrial],
                    W=learned_W_this_trial,
                    M=learned_M_this_trial, 
                    p0_z=learned_p0_z_this_trial,
                    x_oracle=x_oracle,
                    sigma=sigma)
                
                # sufficient stats at each iter is a combination of those from teh previous trial and those from the current iter
                w_table_numer_this_trial = _update(w_table_numer, gamma.sum(0) @ w_arr, lr)
                w_table_denom_this_trial = _update(w_table_denom, gamma.sum((0, -1)), lr)
                tmtrx_this_trial = _update(tmtrx, xi.sum((1, -1)), lr) + eps
            
                learned_M_this_trial = tmtrx_this_trial / tmtrx_this_trial.sum(axis=-1, keepdims=True)
                learned_W_this_trial = w_table_numer_this_trial / (w_table_denom_this_trial[..., None] + eps)
                learned_p0_z_this_trial = _update(learned_p0_z, gamma[..., 0].sum((0, 1)), lr)


            # log suff stats from the final iter as the suff stats for the next trial
            w_table_numer = w_table_numer_this_trial.copy()
            w_table_denom = w_table_denom_this_trial.copy()
            tmtrx = tmtrx_this_trial.copy()

            # online update of parameters
            learned_M = tmtrx / tmtrx.sum(axis=-1, keepdims=True) * lr + learned_M * (1 - lr)
            learned_W = w_table_numer / (w_table_denom[..., None] + eps) * lr + learned_W * (1 - lr)
            learned_p0_z = learned_p0_z_this_trial.copy() * lr + learned_p0_z * (1 - lr)

            if itrial % 10 == 0:
                # evaluate the final parameters on all the trials
                gamma_across_trials, curr_LLpT = get_stats_for_multiple_trials(
                    trials, learned_W, learned_M, learned_p0_z, sigma, x_oracle)

                LLpT_over_time.append(curr_LLpT)


    return learned_M, learned_W, learned_p0_z, LLpT_over_time, gamma_across_trials


def get_stats_for_single_trial(
        single_trial: tuple,
        W: np.ndarray,
        M: np.ndarray,
        p0_z: np.ndarray,
        x_oracle: bool,
        sigma: float):
    """

    Args:
        single_trial: tuple (c, x, sy)
        W: p(w_t|x, z_t=i): shape (nx, nz, d)
        M: p(z_t=i, z_{t+1}=j|c): shape (nc, nz, nz)
        p0_z: p(z_1=i): shape (nz,)
        prior_cx: p(c, x): shape (nc, nx)
        sigma: std of observation noise
    
    Returns:
        gamma: p(c, x, z_t=i| w_{1:T}): shape (nc, nx, nz, nT)
        xi: p(c, x, z_t=i, z_{t+1}=j| w_{1:T}): shape (nc, nx, nz, nz, nT-1)
        p_cx: p(c, x | w_{1:T}): shape (nc, nx)
        LL: log p(w_{1:T}): shape None
    """

    nx, nz, _ = W.shape
    nc = M.shape[0]
    assert M.shape[1:] == (nz, nz)
    assert p0_z.shape == (nz,)
    assert sigma > 0
    
    c, x, sy = single_trial
    obs = np.concatenate((sy['s'], sy['y']), axis=-1)
    
    prior_cx = np.zeros((nc, nx))
    if x_oracle:
        prior_cx[c, x] = 1
    else:
        prior_cx[c] = 1

    prior_cx /= prior_cx.sum()

    x_set = np.arange(nx)
    nT = len(obs)
    log_likes = np.zeros((nx, nz, nT))

    for ix, _ in enumerate(x_set):
        for iz in range(nz):
            log_likes[ix, iz, :] = \
                inference.multivariate_log_likelihood(
                    obs=obs, mean=W[ix, iz][None, :],
                    sigma=sigma)

    gamma, xi, p_cx, LL = inference.forward_backward(
        p0_z, M, log_likes, prior_cx=prior_cx)

    return gamma, xi, p_cx, LL


def get_stats_for_multiple_trials(
        trials,
        W,
        M,
        p0_z,
        sigma,
        x_oracle):
    """
    For the given parameters, compute the LLpT.

    
    Args:
        trials: list of trials with each trial being a tuple (c, x, sy)
        W: p(w_t|x, z_t=i): shape (nx, nz, d)
        M: p(z_t=i, z_{t+1}=j|c): shape (nc, nz, nz)
        p0_z: p(z_1=i): shape (nz,)
        sigma: std of observation noise
    
    Returns:
        gamma_across_trials: list of gamma for each trial
        LLpT: log likelihood of a single trial, averaged over trials
    """
    total_LL = 0
    gamma_across_trials = []
    for itrial in range(len(trials)):

        gamma, _, _, LL = get_stats_for_single_trial(
            single_trial=trials[itrial],
            W=W, M=M, p0_z=p0_z, x_oracle=x_oracle, sigma=sigma)
        total_LL += LL
        gamma_across_trials.append(gamma)
    
    LLpT = total_LL / len(trials)
    
    return gamma_across_trials, LLpT