import numpy as np
import sklearn.cluster
from scipy.special import logsumexp
from task_generation import *
# Terminology: 
# LL: log likelihood of each trial
# total_LL: sum of LLs across trials
# LLpT: log likelihood per trial, total_LL / n_trials


class TaskModel:
    def __init__(self, nc, nz, nx, d, n_epochs_each_task, w_fixate, sigma=0.01):
        self.nc = nc
        self.nz = nz
        self.nx = nx
        self.d = d
        self.sigma = sigma
        self.n_epochs_each_task = n_epochs_each_task
        assert len(n_epochs_each_task) == nc

        # initialize parameters
        self.M = np.ones((nc, nz, nz))
        self.M /= self.M.sum(axis=-1, keepdims=True)
        self.W = np.zeros((nx, nz, d))
        self.p0_z = np.ones(nz)
        self.p0_z /= self.p0_z.sum()

        # initialize sufficient statistics
        self.transition_counts = np.zeros((nc, nz, nz))
        self.W_numer = np.zeros((nx, nz, d))
        self.W_denom = np.zeros((nx, nz))

        # other artifacts for dynamic initialization
        self.familiar_cx = np.zeros((nc, nx))
        self.familiar_xz = np.zeros((nx, nz))
        self.w_fixate = w_fixate.reshape(1, -1)
        assert w_fixate.shape == (1, d)

    def dynamic_initialize_W(self, trial):
        self.W, self.familiar_cx, self.familiar_xz = dynamic_initialize_W(
            trial=trial,
            curr_W=self.W,
            num_epochs_each_task=self.n_epochs_each_task,
            familiar_cx=self.familiar_cx,
            familiar_xz=self.familiar_xz,
            w_fixate=self.w_fixate,
            match_criterion=0.3)
        
    
    def learn_single_trial(self, trial, lr=0.1, suff_stats_discount=0.99, iters_per_trial=10):
        self.M, self.W, self.p0_z, self.W_numer, self.W_denom, self.transition_counts = \
            _online_learning_single_trial(
                curr_W=self.W,
                curr_M=self.M,
                curr_p0_z=self.p0_z,
                curr_W_numer=self.W_numer,
                curr_W_denom=self.W_denom,
                curr_transition_counts=self.transition_counts,
                trial=trial,
                lr=lr,
                suff_stats_discount=suff_stats_discount,
                iters_per_trial=iters_per_trial,
                sigma=self.sigma,
                x_oracle=False,
                eps=1e-10,
                w_update_threshold=1e-3)
        
    
    def infer(self, trial, use_y=True, forward_only=False):
        """
        gamma: p(c, x, z_t=i| w_{1:T}): shape (nc, nx, nz, nT)
        xi: p(c, x, z_t=i, z_{t+1}=j| w_{1:T}): shape (nc, nx, nz, nz, nT-1)
        p_cx: p(c, x | w_{1:T}): shape (nc, nx)
        LL: log p(w_{1:T}): shape None
        """
        gamma, _, _, LL = get_stats_for_single_trial(
            single_trial=trial,
            W=self.W,
            M=self.M,
            p0_z=self.p0_z,
            x_oracle=False,
            sigma=self.sigma,
            show_y=use_y,
            forward_only=forward_only)
        return gamma, LL


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


def _check_for_matching_w(W, w, match_criterion=0.5):
    """
    For a given vector w, check if there is a matching row in W.
    If so, return the indices of the matching rows. 

    Indices are organized as a tuple of arrays. E.g., if W is 3D and 
    both W[3, 5, :] and W[2, 4 , :] match with w, it would return
    (array([3, 2]), array([5, 4])).

    A "match" is defined as a row in W that is within match_criterion of w in
    terms of Euclidean distance.
    """
    # assert W.shape[-1] == len(w)
    dist = np.linalg.norm(W - w, axis=-1)

    matches = np.where(dist < match_criterion)
    return matches


def _get_ordered_clusters(obs: np.array, n_clusters: int):
    """
    For a given sequence of observations (T x D), do k-means clustering and
    and order the clusters by first appearance in the sequence.
    """
    kmeans = sklearn.cluster.KMeans(
        n_clusters=n_clusters,
        random_state=0).fit(obs)

    seen = set()
    ordered_clusters = []
    for l in kmeans.labels_:
        if l not in seen:
            seen.add(l)
            ordered_clusters.append(kmeans.cluster_centers_[l])
    return np.array(ordered_clusters)


def dynamic_initialize_W(trial,
                         curr_W,
                         num_epochs_each_task: np.array,
                         familiar_cx: np.array,
                         familiar_xz: np.array,
                         w_fixate,
                         match_criterion=0.5):
    """
    Initialize W (parameters of the observation models) dynamically for each
    trial.

    Args:
        trial: tuple (c, x, sy)
        curr_W: current W; shape (nx, nz, d)
        num_epochs_each_task: list of number of epochs for each task
        familiar_cx: p(c, x) indicating if a (c, x) pair has been seen before
        familiar_xz: p(x, z) indicating if a (x, z) pair has been seen before
        w_fixate: the fixation/delay epoch for the current task
        match_criterion: distance threshold for matching clusters
    """
    
    # get some dimensions
    nx, nz, d = curr_W.shape
    nc = len(num_epochs_each_task)
    assert familiar_cx.shape == (nc, nx)
    assert familiar_xz.shape == (nx, nz)
    
    # hard code the 0th epoch as the F/D epoch
    curr_W[:, 0, :] = w_fixate
    familiar_xz[:, 0] = 1

    c, _, sy = trial
    obs = np.concatenate([sy['s'], sy['y']], axis=1)

    # do k-means clustering; sort the clusters in order of first appearance.
    # Assuming all epochs have different clusters.
    ordered_clusters = _get_ordered_clusters(obs, num_epochs_each_task[c])

    # remove clusters matching the fixation/delay epoch
    dist_from_w_fixate = np.linalg.norm(ordered_clusters - w_fixate, axis=-1)
    
    non_fixate_clusters = ordered_clusters[dist_from_w_fixate > match_criterion]

    # find the putative x for this trial    
    putative_x = None

    # for each x, how many clusters can be explained by it (under various z)?
    # "explained" as in the cluster matches the w_{x,z} for some z.
    n_clusters_matched_to_this_x = np.zeros(nx)

    for w in non_fixate_clusters:
        matched_xs, matched_zs = _check_for_matching_w(
            curr_W, w, match_criterion)
        n_clusters_matched_to_this_x[matched_xs] += 1

    all_matched = False
    # there exists an x that can explain all clusters
    if np.max(n_clusters_matched_to_this_x) == len(non_fixate_clusters):
        putative_x = np.where(n_clusters_matched_to_this_x == len(non_fixate_clusters))[0][0]
        all_matched = True

    # some of the clusters have matches. Use these to figure out the putative x.
    # To be a possible candidate putative x, it must have some matches and also
    # not seen before.
    elif np.max(n_clusters_matched_to_this_x) > 0:
         
        candidate_putative_x = (n_clusters_matched_to_this_x == np.max(n_clusters_matched_to_this_x))
        candidate_putative_x *= (familiar_cx[c] == 0)
        putative_x = np.where(candidate_putative_x)[0][0]

    # none of the clusters matched anything in W. In which case we arbitrarily
    # assign an unseen x
    else:
        putative_x = np.where(familiar_cx[c] == 0)[0][0]

    # for the assigned putative x, list the unexplained clusters
    if not all_matched:
        unmatched_clusters = []
        for center in non_fixate_clusters:
            (matched_zs, ) = _check_for_matching_w(
                curr_W[putative_x], center, match_criterion)
            if len(matched_zs) == 0:
                unmatched_clusters.append(center)

        
        if len(unmatched_clusters) == 0:
            pass
        else:

            # assign the unmatched clusters to the next available z slots for 
            # the putative x
            available_z_slots = np.where(familiar_xz[putative_x] == 0)[0]

            assert len(available_z_slots) >= len(unmatched_clusters)
            for j, w in enumerate(unmatched_clusters):
                curr_W[putative_x, available_z_slots[j]] = w
                familiar_xz[putative_x, available_z_slots[j]] = 1

            # mark this (c, x) pair as familiar
            familiar_cx[c, putative_x] = 1

        
    curr_W *= familiar_xz[:, :, None]
    return curr_W, familiar_cx, familiar_xz


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
        w_update_threshold: only update w params if the total expected visits of
          this state in the current trial exceeds this value; prevents decay of
          w params on rarely visited states
    
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
        sigma: float,
        show_y=True,
        forward_only=False):
    """

    Args:
        single_trial: tuple (c, x, sy)
        W: p(w_t|x, z_t=i): shape (nx, nz, d)
        M: p(z_t=i, z_{t+1}=j|c): shape (nc, nz, nz)
        p0_z: p(z_1=i): shape (nz,)
        prior_cx: p(c, x): shape (nc, nx)
        sigma: std of observation noise
        show_y: if True, use both s and y for inference; else use only s
        forward_only: if True, only do forward pass; else do both forward and backward pass
    
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
    
    if show_y:
        obs = np.concatenate((sy['s'], sy['y']), axis=-1)
    else:
        obs = sy['s']
    
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
            if show_y:
                log_likes[ix, iz, :] = \
                    multivariate_log_likelihood(
                        obs=obs, mean=W[ix, iz][None, :],
                        sigma=sigma)
            else:
                log_likes[ix, iz, :] = \
                    multivariate_log_likelihood(
                        obs=obs, mean=W[ix, iz][None, :obs.shape[1]],
                        sigma=sigma)

    gamma, xi, p_cx, LL = forward_backward(
        p0_z, M, log_likes, prior_cx=prior_cx, forward_only=forward_only)

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


def _log(x):
    output = x.copy()
    output[x == 0] = -np.inf
    output[x != 0] = np.log(x[x != 0])
    return output


def viterbi(p0_z, transition_matrix, log_likes):
    nc, nz, _ = transition_matrix.shape
    nx, _, nT = log_likes.shape
    scores = np.zeros((nc, nx, nz, nT))
    args = np.zeros((nc, nx, nz, nT), dtype=int)
    for it in range(nT - 2, -1, -1):
        vals = scores[..., it + 1] + log_likes[..., it + 1]
        vals = _log(transition_matrix[:, None, :, :]) + vals[:, :, None, :]
        args[..., it + 1] = np.argmax(vals, axis=-1)
        scores[..., it] = np.max(vals, axis=-1)
    scores_ = _log(p0_z) + scores[..., 0] + log_likes[..., 0]
    ind_ = np.unravel_index(np.argmax(scores_, axis=None), scores_.shape)
    z_ = np.zeros(nT, dtype=int)
    z_[0] = ind_[2]
    for it in range(1, nT):
        z_[it] = args[ind_[0], ind_[1], z_[it - 1], it]
    return ind_, z_


def forward_backward(p0_z, transition_matrix, log_likes, prior_cx=None, forward_only=False):
    nc, nz, _ = transition_matrix.shape
    nx, _, nT = log_likes.shape
    alpha = np.zeros((nc, nx, nz, nT))
    alpha[..., 0] = _log(p0_z) + log_likes[..., 0]
    for it in range(nT - 1):
        m = np.max(alpha[..., it], axis=-1, keepdims=True)
        with np.errstate(divide='ignore'):
            alpha[..., it + 1] = _log(np.einsum('cxi,cij->cxj', np.exp(alpha[..., it] - m),
                                                   transition_matrix)) + m + log_likes[..., it + 1]
    beta = np.zeros((nc, nx, nz, nT))

    if not forward_only:
        beta[..., nT - 1] = 0
        for it in range(nT - 2, -1, -1):
            tmp = log_likes[..., it + 1] + beta[..., it + 1]
            m = np.max(tmp, axis=-1, keepdims=True)
            with np.errstate(divide='ignore'):
                beta[..., it] = _log(np.einsum('cij,cxj->cxi', transition_matrix,
                                                np.exp(tmp - m))) + m

    prior_cx = np.ones((nc, nx)) / (nc * nx) if prior_cx is None else prior_cx
    # logp(w_{1:T}, c, x): shape (nc, nx)
    logp_cx = logsumexp(alpha[..., -1], axis=-1) + _log(prior_cx)
    # logp(w_{1:T}): shape None
    logp_obsv = logsumexp(logp_cx)
    # p(c, x | w_{1:T})
    p_cx = np.exp(logp_cx - logp_obsv)
    # p(z_t=i, c, x | w_{1:T}): shape (nc, nx, nz, nT)

    if forward_only:
        # in this case we want gamma to be p(c_t, z_t, x_t|w{1:t}), but the 
        # expression below is actually p(c_t, z_t, x_t, w{1:t}) and we don't 
        # have p(w{1:t}) in the denominator. So we need to normalize it manually.
        gamma = alpha + beta + _log(prior_cx[:, :, None, None])
        gamma = gamma - logsumexp(gamma, axis=(0, 1, 2), keepdims=True)
        gamma = np.exp(gamma)
    
    else:
        # in this case the normalization factor is p(w{1:T}), which we have,
        # so we can just use the expression below.
        gamma = alpha + beta + _log(prior_cx[:, :, None, None]) - logp_obsv
        gamma = np.exp(gamma)


    # p(z_t=i, z_{t+1}=j, c, x | w_{1:T}): shape (nc, nx, nz, nz, nT-1)
    xi = alpha[:, :, :, None, :-1] + beta[:, :, None, :, 1:] +\
          log_likes[:, None, :, 1:]
    xi += _log(transition_matrix)[:, None, :, :, None] +\
          _log(prior_cx[:, :, None, None, None])

    # using forward_only requires explicit normalization
    if forward_only:
        xi = xi - logsumexp(xi, axis=(0, 1, 2, 3), keepdims=True)
        xi = np.exp(xi)
    else:
        xi = xi - logp_obsv
        xi = np.exp(xi)
    

    return gamma, xi, p_cx, logp_obsv