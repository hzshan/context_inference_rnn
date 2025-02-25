import numpy as np
import inference

# Terminology: 
# LL: log likelihood of each trial
# total_LL: sum of LLs across trials
# LLpT: log likelihood per trial, total_LL / n_trials


def _update(orig, new, lr, mask=None):
    if mask is None:
        return orig * (1 - lr) + new * lr
    else:
        output = orig.copy()
        output[mask] = output[mask] * (1 - lr) + new[mask] * lr
        return output


def _add_with_discount(orig, new, discount):
    return orig * discount + new


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
        transition_counts = np.zeros((nc, nz, nz))
        W_numer = np.zeros((nx, nz, 6))  # 6 is the dimension of stim + response
        W_denom = np.zeros((nx, nz))
        gamma_across_trials = []

        for itrial in range(n_trials):
            _, _, sy = trials[itrial]
            w_arr = np.concatenate((sy['s'], sy['y']), axis=-1)

            gamma, xi, p_cx, LL = get_stats_for_single_trial(
                single_trial=trials[itrial],
                W=learned_W,
                M=learned_M,
                p0_z=learned_p0_z,
                x_oracle=x_oracle,
                sigma=sigma)
            
            gamma_across_trials.append(gamma)

            # nc, nx, nz, nT; # nc, nx, nz, nz, nT-1; # nc, nx;
            curr_total_LL += LL

            learned_p0_z += gamma[..., 0].sum((0, 1))
            transition_counts += xi.sum((1, -1))
            W_numer += gamma.sum(0) @ w_arr  # nx, nz, d
            W_denom += gamma.sum((0, -1))

        # check convergence
        LLpT_over_time.append(curr_total_LL / n_trials)
        if len(LLpT_over_time) > 2:
            if np.abs(LLpT_over_time[-1] - LLpT_over_time[-2]) < tol:
                break
        
        # update parameters
        learned_p0_z += eps
        learned_p0_z = learned_p0_z / learned_p0_z.sum()
        transition_counts += eps
        learned_M = transition_counts / transition_counts.sum(axis=-1, keepdims=True)
        learned_W = W_numer / (W_denom[..., None] + eps)
    
    # warn if not converged
    if irun == max_iter - 1:
        print('Warning: EM did not converge')

    return learned_M, learned_W, learned_p0_z, LLpT_over_time, gamma_across_trials


def online_learning(init_M,
                    init_W,
                    init_p0_z,
                    trials,
                    n_sweeps=1,
                    sigma=0.01,
                    lr=0.1,
                    suff_stats_discount=0.99,
                    iters_per_trial=10,
                    x_oracle=False,
                    eps=1e-10,
                    w_update_threshold=1e-3):
    """
    Args:
        init_M: initial transition matrix p(z_t=i, z_{t+1}=j|c): shape (nc, nz, nz)
        init_W: initial weight table p(w_t|x, z_t=i): shape (nx, nz, d)
        init_p0_z: initial distribution of z p(z_1=i): shape (nz,)
        trials: list of trials with each trial being a tuple (c, x, sy)
        n_sweeps: number of sweeps (for true online learning, set to 1)
        sigma: std of observation noise
        lr: learning rate
        suff_stats_discount: discount factor for sufficient statistics
        iters_per_trial: number of iterations per trial
        eps: small constant for numerical stability
        w_update_threshold: only update w params if the total expected visits of this state in the current trial exceeds this value; prevents decay of w params on rarely visited states
    
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
    transition_counts = np.zeros((nc, nz, nz))
    W_numer = np.zeros((nx, nz, 6))  # 6 is the dimension of stim + response
    W_denom = np.zeros((nx, nz))

    for i in range(n_sweeps):
        
        for itrial in range(len(trials)):
            
            (learned_M, learned_W,
             learned_p0_z, W_numer,
             W_denom, transition_counts) = _online_learning_single_trial(
                curr_W=learned_W,
                curr_M=learned_M,
                curr_p0_z=learned_p0_z,
                curr_W_numer=W_numer,
                curr_W_denom=W_denom,
                curr_transition_counts=transition_counts,
                trial=trials[itrial],
                lr=lr,
                suff_stats_discount=suff_stats_discount,
                iters_per_trial=iters_per_trial,
                sigma=sigma,
                x_oracle=x_oracle,
                eps=eps,
                w_update_threshold=w_update_threshold)

            if itrial % 10 == 0:
                # evaluate the final parameters on all the trials
                gamma_across_trials, curr_LLpT, _ = get_stats_for_multiple_trials(
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
        LL_across_trials: list of log likelihood for each trial
    """

    gamma_across_trials = []
    LL_across_trials = []
    for itrial in range(len(trials)):

        gamma, _, _, LL = get_stats_for_single_trial(
            single_trial=trials[itrial],
            W=W, M=M, p0_z=p0_z, x_oracle=x_oracle, sigma=sigma)
        LL_across_trials.append(LL)
        gamma_across_trials.append(gamma)

    total_LL = np.sum(LL_across_trials)
    LLpT = total_LL / len(trials)
    
    return gamma_across_trials, LLpT, LL_across_trials


def _online_learning_single_trial(
    curr_W,
    curr_M,
    curr_p0_z,
    curr_W_numer,
    curr_W_denom,
    curr_transition_counts,
    trial,
    lr,
    suff_stats_discount,
    iters_per_trial,
    sigma,
    x_oracle,
    w_update_threshold=1e-3,
    eps=1e-10):

    c, x, sy = trial
    w_arr = np.concatenate((sy['s'], sy['y']), axis=-1)

    learned_W_this_trial = curr_W.copy()
    learned_M_this_trial = curr_M.copy()
    learned_p0_z_this_trial = curr_p0_z.copy()

    W_numer_this_trial = None
    W_denom_this_trial = None
    transition_counts_this_trial = None

    for iiter in range(iters_per_trial):

        # get sufficient statistics for this trial
        gamma, xi, _, _ = get_stats_for_single_trial(
            single_trial=trial,
            W=learned_W_this_trial,
            M=learned_M_this_trial, 
            p0_z=learned_p0_z_this_trial,
            x_oracle=x_oracle,
            sigma=sigma)
        
        # sufficient stats at each iter is a combination of those from the
        #  previous trial and those from the current iter
        W_numer_this_trial = _add_with_discount(
            curr_W_numer, gamma.sum(0) @ w_arr, suff_stats_discount)
        W_denom_this_trial = _add_with_discount(
            curr_W_denom, gamma.sum((0, -1)), suff_stats_discount)
        transition_counts_this_trial = _add_with_discount(
            curr_transition_counts, xi.sum((1, -1)), suff_stats_discount) + eps

        # only decay/update parts of W that are visited a lot
        W_update_mask = np.repeat(
            W_denom_this_trial[..., None] > w_update_threshold, 6, axis=-1)

        # update parameters
        learned_M_this_trial = _update(
            curr_M,
            (transition_counts_this_trial /
             transition_counts_this_trial.sum(axis=-1, keepdims=True)),
            lr)
        learned_W_this_trial = _update(
            curr_W,
            W_numer_this_trial / (W_denom_this_trial[..., None] + eps),
            lr,
            mask=W_update_mask)
        learned_p0_z_this_trial = _update(
            curr_p0_z,
            gamma[..., 0].sum((0, 1)),
            lr)

    # store stats/params after this trial
    W_numer = W_numer_this_trial.copy()
    W_denom = W_denom_this_trial.copy()
    transition_counts = transition_counts_this_trial.copy()

    learned_M = learned_M_this_trial.copy()
    learned_W = learned_W_this_trial.copy()
    learned_p0_z = learned_p0_z_this_trial.copy()

    return learned_M, learned_W, learned_p0_z, W_numer, W_denom, transition_counts