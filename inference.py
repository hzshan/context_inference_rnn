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
        if c_cur != c_pre:
            x = np.random.choice(x_set)
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

    ###########
    # given s, y, cue, infer c, z, x
    p0_cz = np.zeros(len(cz_list))
    p0_cz[cz_list.index((1, 0))] = 0.5
    p0_cz[cz_list.index((-1, 0))] = 0.5
    p0_cz += 1e-10
    p0_cz /= p0_cz.sum()
    p0_x = np.arange(len(x_set)) / len(x_set)

    q_czt_arr = np.zeros((len(s_arr), len(cz_list)))
    q_xt_arr = np.zeros((len(s_arr), len(x_set)))
    c_infer_arr = np.zeros(len(s_arr))
    z_infer_arr = np.zeros(len(s_arr))
    x_infer_arr = np.zeros(len(s_arr))
    for it in range(len(s_arr)):
        s_t = s_arr[it]
        y_t = y_arr[it]
        cue_t = cue_arr[it]
        logp_collect = np.zeros((len(cz_list), len(x_set)))
        for icz, (c, z) in enumerate(cz_list):
            for ix, x in enumerate(x_set):
                logp_collect[icz, ix] = compute_emission_logp(s_t, y_t, z, x, sigma, sigma)
                logp_collect[icz, ix] += compute_cue_logp(cue_t, c, z, sigma)
        if it == 0:
            q_xt = p0_x
            alpha_cz = np.log(p0_cz) + logp_collect @ q_xt
            q_czt = np.exp(alpha_cz - logsumexp(alpha_cz))
        else:
            q_xt = q_xt_arr[it - 1]
            for iter in range(10):
                with np.errstate(divide="ignore"):
                    alpha_cz = np.log(q_czt_arr[it - 1] @ transition_mtrx) + logp_collect @ q_xt
                q_czt = np.exp(alpha_cz - logsumexp(alpha_cz))
                alpha_x = q_czt @ logp_collect
                q_xt = np.exp(alpha_x - logsumexp(alpha_x))

        q_xt_arr[it] = q_xt
        q_czt_arr[it] = q_czt
        c_infer, z_infer = cz_list[np.argmax(q_czt)]
        c_infer_arr[it] = c_infer
        z_infer_arr[it] = z_infer
        x_infer_arr[it] = 0 if q_xt[0] == q_xt[1] else x_set[np.argmax(q_xt)]


    ########################
    fig, axes = plt.subplots(8, 1, figsize=(7, 7), sharex=True)
    axes[0].plot(c_arr)
    axes[0].plot(c_infer_arr, alpha=0.5)
    axes[0].set_ylabel('c')
    axes[1].plot(z_arr)
    axes[1].plot(z_infer_arr, alpha=0.5)
    axes[1].set_ylabel('z')
    axes[2].plot(x_arr)
    axes[2].plot(x_infer_arr, alpha=0.5)
    axes[2].set_ylabel('x')
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