import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from scipy.special import logsumexp


def compute_emission_mu(z_t, x):
    if z_t in [0, 3]: # fixation or delay
        mu_s = np.array([0, 1])
        mu_y = np.array([0, 1])
    elif z_t == 1: # stim-pro
        mu_s = np.array([x, 1])
        mu_y = np.array([0, 1])
    elif z_t == 2: # stim-anti
        mu_s = np.array([-x, 1])
        mu_y = np.array([0, 1])
    elif z_t == 4: # response
        mu_s = np.array([0, 0])
        mu_y = np.array([x, 0])
    else:
        raise NotImplemented
    return mu_s, mu_y


def compute_emission_logp(s_t, y_t, z_t, x, sigma_s=0.1, sigma_y=0.1):
    mu_s, mu_y = compute_emission_mu(z_t, x)
    logp = multivariate_normal.logpdf(s_t, mean=mu_s, cov=np.square(sigma_s)*np.eye(2))
    logp += multivariate_normal.logpdf(y_t, mean=mu_y, cov=np.square(sigma_y)*np.eye(2))
    return logp


def sample_emission(z_t, x, sigma_s=0.1, sigma_y=0.1):
    mu_s, mu_y = compute_emission_mu(z_t, x)
    s_t = np.random.normal(mu_s, sigma_s)
    y_t = np.random.normal(mu_y, sigma_y)
    return s_t, y_t


def compute_cue_mu(c, z_t):
    mu_cue = c if z_t == 0 else 0
    return mu_cue


def compute_cue_logp(cue_t, c, z_t, sigma_cue=0.1):
    mu_cue = compute_cue_mu(c, z_t)
    logp = multivariate_normal.logpdf(cue_t, mean=mu_cue, cov=np.square(sigma_cue)*np.eye(1))
    return logp


def sample_cue(c, z_t, sigma_cue=0.1):
    mu_cue = compute_cue_mu(c, z_t)
    return np.random.normal(mu_cue, sigma_cue)


def expanded_transition_prob(c_pre, z_pre, c_cur, z_cur, p_stay=0.9):
    # [0: fixation, 1: stim-pro, 2: stim-anti, 3: delay, 4: response]
    # pro-0, pro-1, pro-3, pro-4, anti-0, anti-2, anti-3, anti-4
    pro_z_transition = {(0, 0): p_stay, (0, 1): 1 - p_stay,
                        (1, 1): p_stay, (1, 3): (1 - p_stay) / 2, (1, 4): (1 - p_stay) / 2,
                        (3, 3): p_stay, (3, 4): 1 - p_stay,
                        (4, 4): p_stay}
    anti_z_transition = {(0, 0): p_stay, (0, 2): 1 - p_stay,
                         (2, 2): p_stay, (2, 3): (1 - p_stay) / 2, (2, 4): (1 - p_stay) / 2,
                         (3, 3): p_stay, (3, 4): 1 - p_stay,
                         (4, 4): p_stay}
    if c_pre is None and z_cur == 0: # time step 0
        return 0.5
    elif c_pre == 1 and c_cur == 1 and (z_pre, z_cur) in pro_z_transition:  # pro
        return pro_z_transition[(z_pre, z_cur)]
    elif c_pre == -1 and c_cur == -1 and (z_pre, z_cur) in anti_z_transition:  # anti
        return anti_z_transition[(z_pre, z_cur)]
    elif c_pre == 1 and z_pre == 4 and c_cur == -1 and z_cur == 0:  # pro response -> anti fixation
        return 1 - p_stay
    elif c_pre == -1 and z_pre == 4 and c_cur == 1 and z_cur == 0:  # anti response -> pro fixation
        return 1 - p_stay
    else:
        return 0


if __name__ == '__main__':

    ######################### task #####################
    p_stay = 0.9
    sigma = 0.1

    cz_list = [(1, z) for z in [0, 1, 3, 4]] + [(-1, z) for z in [0, 2, 3, 4]]
    x_set = [-1, 1]
    transition_mtrx = np.zeros((len(cz_list), len(cz_list)))
    for i, (c_pre, z_pre) in enumerate(cz_list):
        for j, (c_cur, z_cur) in enumerate(cz_list):
            transition_mtrx[i, j] = expanded_transition_prob(c_pre, z_pre, c_cur, z_cur, p_stay=p_stay)

    i_pre, c_pre, z_pre = None, None, None
    x = np.random.choice(x_set)
    c_arr, z_arr, x_arr, s_arr, y_arr, cue_arr = [], [], [], [], [], []
    for it in range(200):
        if i_pre is None:
            i_cur = np.random.choice([cz_list.index((1, 0)), cz_list.index((-1, 0))])
        else:
            i_cur = np.random.choice(len(cz_list), p=transition_mtrx[i_pre])
        c_cur, z_cur = cz_list[i_cur]
        # if c_cur != c_pre:
        #     x = np.random.choice(x_set)
        s, y = sample_emission(z_cur, x, sigma_s=sigma, sigma_y=sigma)
        cue = sample_cue(c_cur, z_cur, sigma_cue=sigma)
        c_arr.append(c_cur)
        z_arr.append(z_cur)
        x_arr.append(x)
        s_arr.append(s)
        y_arr.append(y)
        cue_arr.append(cue)
        i_pre, c_pre, z_pre = i_cur, c_cur, z_cur
    c_arr, z_arr, x_arr = np.array(c_arr), np.array(z_arr), np.array(x_arr)
    s_arr, y_arr, cue_arr = np.array(s_arr), np.array(y_arr), np.array(cue_arr)

    ##################### given s, y, cue, infer c, z, x  ####################
    nT = len(s_arr)
    log_likes = np.zeros((nT, len(x_set), len(cz_list)))
    for it in range(nT):
        for ix, x in enumerate(x_set):
            for icz, (c, z) in enumerate(cz_list):
                log_likes[it, ix, icz] = compute_emission_logp(s_arr[it], y_arr[it], z, x, sigma, sigma)
                log_likes[it, ix, icz] += compute_cue_logp(cue_arr[it], c, z, sigma)

    log_eps = 1e-10
    p0_cz = np.zeros(len(cz_list))
    p0_cz[cz_list.index((1, 0))] = 0.5
    p0_cz[cz_list.index((-1, 0))] = 0.5
    p0_cz += log_eps
    p0_cz /= p0_cz.sum()

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

    ####################### viterbi ##############################
    scores = np.zeros((nT, len(x_set), len(cz_list)))
    args = np.zeros((nT, len(x_set), len(cz_list)), dtype=int)
    for it in range(nT - 2, -1, -1):
        vals = (scores[it + 1] + log_likes[it + 1]).T
        vals = np.log(transition_mtrx[:, :, None] + log_eps) + vals[None, :, :]
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
        cz_[it] = args[it, ind_[0], int(cz_[it - 1])]
    cz_ = np.array([cz_list[cur] for cur in cz_])

    ######################## plotting #######################
    fig, axes = plt.subplots(8, 1, figsize=(7, 10), sharex=True)
    axes[0].plot(c_arr, label='ground truth')
    axes[0].plot(cz_[:, 0], alpha=0.5, label='Viterbi')
    axes[0].set_ylabel('c')
    axes[0].legend(bbox_to_anchor=(1.05, 0.5), loc='center')
    axes[1].plot(z_arr, label='ground truth')
    axes[1].plot(cz_[:, 1], alpha=0.5, label='Viterbi')
    axes[1].set_ylabel('z')
    axes[1].legend(bbox_to_anchor=(1.05, 0.5), loc='center')
    axes[2].plot(x_arr, label='ground truth')
    axes[2].plot([x_] * nT, alpha=0.5, label='Viterbi')
    axes[2].set_ylabel('x')
    axes[2].legend(bbox_to_anchor=(1.05, 0.5), loc='center')
    axes[3].plot(s_arr[:, 0])
    axes[3].set_ylabel('s[0]')
    axes[4].plot(s_arr[:, 1])
    axes[4].set_ylabel('s[1]')
    axes[5].plot(y_arr[:, 0])
    axes[5].set_ylabel('y[0]')
    axes[6].plot(y_arr[:, 1])
    axes[6].set_ylabel('y[1]')
    axes[7].plot(cue_arr)
    axes[7].set_ylabel('cue')
    fig.show()
    fig.savefig('./figures/example_inference.png', bbox_inches='tight', transparent=True)