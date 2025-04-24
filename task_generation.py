import numpy as np

"""
Notations:
Te: duration of epoch
Tt: duration of trial
sig_s: input noise standard deviation
sig_y: target output noise standard deviation

x: dictionary containing the task variables

# for a dict of epoch types, search for `epoch_dict` (it's defined mid-file)
"""

task_dict = {
    'PRO_D': 'F/D->S->R_P', # "DelayPro": respond to the stimulus once fixation is off
    'PRO_M': 'F/D->S->F/D->R_M_P', # "MemoryPro" respond to the stimulus after a memory period
    'PRO_R': 'F/D->R_P', # "ResponsePro" respond to the stimulus ASAP
    'ANTI_D': 'F/D->S->R_A', # "DelayAnti": respond to the opposite once fixation is off
    'ANTI_M': 'F/D->S->F/D->R_M_A', # "MemoryAnti": respond to the opposite after memory period
    'ANTI_R': 'F/D->R_A', # "ResponseAnti": respond to the opposite ASAP
    'ORDER': 'F/D->S->S_B->R_M_P', # respond to the stimulus coming on first
    'DM': 'F/D->S_S->R_M_P' # respond to the stronger stimulus
}


def make_stim_input(stim: int, Te: int, d_stim: float):
    """
    Produce a Te x 2 matrix of noisy inputs for a given stimulus.
    if stim = -1, return zero-mean random noise

    Args:
        stim: int in [-1, 0, 1, 2, ...]
        duration: number of time steps
        d_stim: if stim >= 0, angle = d_stim * stim
    Returns:
        mean_stim: Te x 2 matrix of noisy inputs

    """
    # assert stim in [-1, 0, 1]

    if stim == -1:
        return np.zeros((Te, 2))
    else:
        angle = d_stim * stim
        mean_stim = np.repeat(np.array([[np.cos(angle), np.sin(angle)]]), Te, axis=0)
        return mean_stim


def make_input(stim: int, fixation: int, Te: int, sig_s: float, d_stim: float,
               contrast=1, other_stim_contrast=0):
    """
    Generates the input (s) for a given epoch. Assuming a single modality.

    Produces a Te x 3 matrix of noisy inputs for a given stimulus pair and
    fixation cue. The first two columns are the inputs for the first stimulus,
    the second two columns are the inputs for the second stimulus, and the
    last column is the fixation cue.

    Args:
        stim: int in [-1, 0, 1]. -1 means no stimulus presentation
        fixation: int in [0, 1]
        Te: number of time steps
        sig_s: noise standard deviation for input
        contrast: stimulus contrast
        other_stim_contrast: contrast of the other stimulus when superimposed

    Returns:
        mean_input: Te x 3 matrix of noisy inputs
    """
    # assert stim in [-1, 0, 1]

    stim_input = make_stim_input(stim, Te, d_stim) * contrast

    if stim != -1:
        stim_input += make_stim_input(1 - stim, Te, d_stim) * other_stim_contrast

    assert fixation in [0, 1]
    fixation_input = np.random.normal(fixation, sig_s, (Te, 1))

    mean_input = np.hstack([stim_input, fixation_input])
    return np.random.normal(mean_input, sig_s)


# def make_bimodal_input(mod1_stim: int, mod2_stim: int,
#                        fixation: int, Te: int, sig_s: float, superimpose=False):
#     """
#     Generates the input (s) for a given epoch.

#     Produces a Te x 3 matrix of noisy inputs for a given stimulus pair and
#     fixation cue. The first two columns are the inputs for the first stimulus,
#     the second two columns are the inputs for the second stimulus, and the
#     last column is the fixation cue.
#     """
#     assert mod1_stim in [-1, 0, 1]
#     assert mod2_stim in [-1, 0, 1]

#     mod1_stim_input = make_stim_input(mod1_stim, Te)
#     mod2_stim_input = make_stim_input(mod2_stim, Te)

#     if superimpose:
#         # in each modality, superimpose the stimulus with the opposite one
#         if mod1_stim != -1:
#             mod1_stim_input += make_stim_input(1 - mod1_stim, Te) * 0.5
#         elif mod2_stim != -1:
#             mod2_stim_input += make_stim_input(1 - mod2_stim, Te) * 0.5

#     assert fixation in [0, 1]
#     fixation_input = np.random.normal(fixation, sig_s, (Te, 1))

#     mean_input = np.hstack([mod1_stim_input, mod2_stim_input, fixation_input]).T
#     return np.random.normal(mean_input, sig_s)


def make_target_output(response: int, fixation: int, Te: int, sig_y: float, d_stim: float):
    """
    Generates the target output (y) for a given epoch.

    Args:
        response: int in [-1, 0, 1]. -1 means no response
        fixation: int in [0, 1]
        Te: number of time steps
        sig_y: noise standard deviation for target output

    Returns:
        mean_output: Te x 3 matrix of noisy target outputs
    """
    target_output = make_stim_input(stim=response, Te=Te, d_stim=d_stim)
    fixation_output = np.ones((Te, 1)) * fixation
    return np.random.normal(
        np.hstack([target_output, fixation_output]), sig_y)


def make_delay_or_fixation_epoch(x: dict, Te: int, sig_s: float, sig_y: float, d_stim: float):
    """
    Generates a fixation/delay epoch (F/D).
    Composed of input (s, Te x 3) and target output (y, Te x 3).
    """

    return {'s':make_input(stim=-1, # no stimulus presentation
                           fixation=1,
                           Te=Te,
                           sig_s=sig_s,
                           d_stim=d_stim),
            'y':make_target_output(response=-1,  # no stimulus response
                                   fixation=1,
                                   Te=Te,
                                   sig_y=sig_y,
                                   d_stim=d_stim)}


def make_response_from_memory_pro_epoch(x: dict, Te: int, sig_s: float, sig_y: float, d_stim: float):
    """
    Generates a response-from-memory pro epoch (R_M_P).
    Composed of input (s, Te x 3) and target output (y, Te x 3).
    """

    return {'s':make_input(stim=-1, # no stimulus presentation
                           fixation=0,
                           Te=Te,
                           sig_s=sig_s,
                           d_stim=d_stim),
            'y':make_target_output(response=x['theta_task'],  #respond to the task variable
                                   fixation=0,
                                   Te=Te,
                                   sig_y=sig_y,
                                   d_stim=d_stim)}


def make_response_from_memory_anti_epoch(x: dict, Te: int, sig_s: float, sig_y: float, d_stim: float):
    """
    Generates a response-from-memory anti epoch (R_M_A).
    Composed of input (s, Te x 3) and target output (y, Te x 3).
    """

    return {'s':make_input(stim=-1, # no stimulus presentation
                           fixation=0,
                           Te=Te,
                           sig_s=sig_s,
                           d_stim=d_stim),
            'y':make_target_output(response=x['theta_task'] + int(np.pi / d_stim),  #respond to the anti task variable
                                   fixation=0,
                                   Te=Te,
                                   sig_y=sig_y,
                                   d_stim=d_stim)}


def make_response_pro_epoch(x: dict, Te: int, sig_s: float, sig_y: float, d_stim: float):
    """
    Generates a response epoch that responds to the presented stimulus. (R_P).
    Composed of input (s, Te x 3) and target output (y, Te x 3).
    """

    return {'s':make_input(stim=x['theta_task'], # no stimulus presentation
                           fixation=0,
                           Te=Te,
                           sig_s=sig_s,
                           d_stim=d_stim),
            'y':make_target_output(response=x['theta_task'],  #respond to the task variable
                                   fixation=0,
                                   Te=Te,
                                   sig_y=sig_y,
                                   d_stim=d_stim)}


def make_response_anti_epoch(x: dict, Te: int, sig_s: float, sig_y: float, d_stim: float):
    """
    Generates a response epoch that responds to the anti stimulus. (R_A).
    Composed of input (s, Te x 3) and target output (y, Te x 3).
    """

    return {'s':make_input(stim=x['theta_task'],
                           fixation=0,
                           Te=Te,
                           sig_s=sig_s,
                           d_stim=d_stim),
            'y':make_target_output(response=x['theta_task'] + int(np.pi / d_stim),  #respond to the anti task variable
                                   fixation=0,
                                   Te=Te,
                                   sig_y=sig_y,
                                   d_stim=d_stim)}


def make_stim_epoch(x: dict, Te: int, sig_s: float, sig_y: float, d_stim: float):
    """
    Generates a stimulus epoch (S).
    Composed of input (s, Te x 3) and target output (y, Te x 3).
    """

    # randomly choose a modality to present the stimulus

    return {'s':make_input(x['theta_task'],
                           fixation=1, Te=Te,
                           sig_s=sig_s, d_stim=d_stim),
            'y':make_target_output(response=-1,
                                   fixation=1,
                                   Te=Te,
                                   sig_y=sig_y, d_stim=d_stim)}


def make_superimposed_epoch(x: dict, Te: int, sig_s: float, sig_y: float, d_stim: float):
    """
    Generates a superimposed stimulus epoch (S_S).
    Composed of input (s, Te x 3) and target output (y, Te x 3).
    """

    return {'s':make_input(stim=x['theta_task'],
                           fixation=1, Te=Te,
                           sig_s=sig_s, d_stim=d_stim, other_stim_contrast=0.5),
            'y':make_target_output(response=-1,
                                  fixation=1,
                                  Te=Te,
                                  sig_y=sig_y, d_stim=d_stim)}


def make_both_epoch(x: dict, Te: int, sig_s: float, sig_y: float, d_stim: float):
    """
    Generates a both-stimuli-on epoch (S_B).
    Composed of input (s, Te x 3) and target output (y, Te x 3).
    """

    return {'s':make_input(stim=x['theta_task'],
                          fixation=1, Te=Te,
                          sig_s=sig_s, d_stim=d_stim,
                          other_stim_contrast=1),
            'y':make_target_output(response=-1,
                                  fixation=1,
                                  Te=Te,
                                  sig_y=sig_y, d_stim=d_stim)}


epoch_dict = {
    'F/D': make_delay_or_fixation_epoch,
    'S': make_stim_epoch,
    'S_B': make_both_epoch,
    'S_S': make_superimposed_epoch,
    'R_M_P': make_response_from_memory_pro_epoch,
    'R_M_A': make_response_from_memory_anti_epoch,
    'R_P': make_response_pro_epoch,
    'R_A': make_response_anti_epoch,
    'F': make_delay_or_fixation_epoch,
    'D': make_delay_or_fixation_epoch,
}


def parse_z_t(epoch_symbol: str, x: dict, sig_s: float, sig_y: float, d_stim: float, Te: int):
    """

    Args:
        epoch_symbol: a string representing the epoch
        x: dictionary containing the task variables
        sig_s: noise standard deviation for input
        sig_y: noise standard deviation for target output
        Te: duration of epoch in number of timesteps

    Returns:
        epoch: a dictionary containing the input (s) and target output (y) of the epoch
    """

    fn_to_call = epoch_dict[epoch_symbol]
    return fn_to_call(x, Te=Te, sig_s=sig_s, sig_y=sig_y, d_stim=d_stim)


def compose_trial(seq_of_epochs: str,
                  x: dict,
                  sig_s: float,
                  sig_y: float,
                  p_stay,
                  min_Te,
                  d_stim=np.pi/2):
    """
    Generates a trial composed of a sequence of epochs.

    seq_of_epochs: a sequence of epoch symbols separated by '->'
    x: the task variable (0 or 1)
    sig_s: noise standard deviation for input
    sig_y: noise standard deviation for target output
    p_stay: probability of staying in the current epoch at each timestep
    min_Te: minimum duration of each epoch

    Returns:
    trial: a dictionary containing the input (s) and target output (y) of the trial
    boundaries: a list of boundaries (in units of timesteps) between epochs
    """
    epoch_symbols = seq_of_epochs.split('->')

    if p_stay is not None:
        transition_probs = np.ones(len(epoch_symbols)) * (1 - p_stay)
        all_Te = [np.max([np.random.geometric(prob), min_Te]) for prob in transition_probs]
    else:
        all_Te = np.random.choice([10, 20, 40, 80], size=len(epoch_symbols))

    epochs = []
    for Te, epoch_symbol in zip(all_Te, epoch_symbols):
        epochs.append(parse_z_t(epoch_symbol, x, sig_s, sig_y, d_stim, Te))
    
    boundaries = np.cumsum(all_Te)
    return merge_epochs(epochs), boundaries


def merge_epochs(epochs):
    """
    Join epochs into a single trial.

    Args:
        epochs: a list of dictionaries containing the input (s) and target output (y) of each epoch
    
    Returns:
        trial: a dictionary containing the input (s) and target output (y) of the trial
    """
    return {'s':np.vstack([epoch['s'] for epoch in epochs]),
            'y':np.vstack([epoch['y'] for epoch in epochs])}


def multivariate_log_likelihood(obs, mean, sigma):
    """
    computes the negative log likelihood of a multivariate normal distribution,
    given the observations x, the means and the standard deviation sigma
    (assuming the covariance matrix is identity times sigma**2)

    Args:
        obs: the observations, [P x N] where P is the number of samples and N the number of dimensions
        mean: the mean [P x N], where each row is the mean of the corresponding sample
        sigma (float): the standard deviation
    
    Returns:
        log_likelihood: [P]
    """
    if obs.ndim == 1:
        obs = obs.reshape(1, -1)
    if mean.ndim == 1:
        mean = mean.reshape(1, -1)

    # assert obs.shape == mean.shape

    k = obs.shape[1]

    sq_dist = np.sum((obs - mean) ** 2, axis=1)
    return -np.log(2 * np.pi * sigma**2) * k / 2 - sq_dist / (2 * sigma**2)


def compute_emission_logp(s_t,
                          y_t,
                          z_t: str,
                          x: dict,
                          sigma_s=0.1,
                          sigma_y=0.1,
                          d_stim=np.pi/2):
    """
    Compute log likelihood of observing (s_t, y_t) under a given epoch (z) and 
    task variable (x) for each t in 1,...,s_t.shape[0].

    Args:
        s_t: [s_t.shape[0] x 3] matrix of observed inputs (Te can be 1)
        y_t: [s_t.shape[0] x 3] matrix of observed outputs
        z_t: string representing the epoch
        x: dictionary containing the task variables
        sigma_s: noise standard deviation for input
        sigma_y: noise standard deviation for target output
    
    Returns:
        logp: [s_t.shape[0]] the log likelihood of observing (s_t, y_t) for each
        each t in 1,...,s_t.shape[0].
    """
    
    assert s_t.shape[0] == y_t.shape[0]

    num_timesteps = s_t.shape[0]

    epoch_mean = parse_z_t(epoch_symbol=z_t, x=x,
                           sig_s=0, sig_y=0, d_stim=d_stim, Te=num_timesteps)

    logp = multivariate_log_likelihood(
        obs=s_t, mean=epoch_mean['s'], sigma=sigma_s)
    logp += multivariate_log_likelihood(
        obs=y_t, mean=epoch_mean['y'], sigma=sigma_y)
    
    return logp


def get_c_z_transition_matrix(task_dict, p_stay):
    """
    Generate the transition matrix for the task epochs. Each state is a 
    c, z tuple.

    Args:
        task_dict: a dictionary where the keys are task types (e.g. 'PRO_D')
        and the corresponding values are sequences of epochs (e.g. 'F->S_P->R_P')
        p_stay: the probability of staying in the same epoch

    Returns:
        transition_matrix: [N x N] matrix where N is the number of epochs
        cz_pairs: a list of all possible c, z pairs
        cz_to_ind_hash: a dictionary mapping c, z pairs to indices
    """
    
    # get a list of all possible c, z pairs
    cz_pairs = []
    for task_type in task_dict.keys():
        epochs = task_dict[task_type].split('->')
        for epoch_type in epochs:
            cz_pairs.append((task_type, epoch_type))
        # cz_pairs.append((task_type, '<EOT>'))

    transition_matrix = np.eye((len(cz_pairs))) * p_stay

    for i, cz_pair in enumerate(cz_pairs):
        epoch_type = cz_pair[1]
        # if epoch_type != '<EOT>':
        if 'R_' not in epoch_type:
            transition_matrix[i, i+1] = 1 - p_stay
        else:
            transition_matrix[i, i] = 1

    cz_to_ind_hash = {c_z_pair: i for i, c_z_pair in enumerate(cz_pairs)}
    
    return transition_matrix, cz_pairs, cz_to_ind_hash


def get_z_transition_matrix_per_task(task_dict, p_stay):
    task_list = list(task_dict.keys())
    z_list = list(epoch_dict.keys())[:-2]
    transition_matrix = np.zeros((len(task_list), len(z_list), len(z_list)))
    for itask, task_type in enumerate(task_list):
        transition_matrix[itask] = np.eye(len(z_list))
        epochs = task_dict[task_type].split('->')
        for iepoch, epoch_type in enumerate(epochs[:-1]):
            iz = z_list.index(epoch_type)
            iz_nxt = z_list.index(epochs[iepoch+1])
            
            transition_matrix[itask][iz, iz] = p_stay
            transition_matrix[itask][iz, iz_nxt] = 1 - p_stay
    return transition_matrix, task_list, z_list


def generate_trials(task_list: list,
                    n_trials: int,
                    nx: int,
                    sigma: float,
                    p_stay: float,
                    min_Te: int,
                    d_stim=None,
                    trial_order='interleaved'):
    
    #TODO: add options to change ordering of trials

    if d_stim is None:
        if nx == 2:
            d_stim = np.pi / 2
        else:
            d_stim = 2 * np.pi / nx

    nc = len(task_list)
    epoch_list = []
    for task in task_list:
        epoch_list += task_dict[task].split('->')
    epoch_list = list(set(epoch_list))

    n_trials_per_task = int(n_trials / nc)
    
    if trial_order == 'interleaved':
        task_ids = np.mod(np.arange(n_trials), nc)
    else:
        assert trial_order == 'blocked'
        task_ids = np.repeat(np.arange(nc), n_trials_per_task)

    trials = []
    
    for itrial in range(n_trials):
        c = task_ids[itrial]
        x = itrial % nx
        sy, _ = compose_trial(
            task_dict[task_list[c]],
            {'theta_task': x}, sigma, sigma, p_stay=p_stay, min_Te=min_Te, d_stim=d_stim)
        trials.append((c, x, sy))

    trials = np.array(trials)

    return trials, epoch_list


def get_ground_truth(task_list, epoch_list, p_stay, nx, d_stim=None, eps=1e-10):

    if d_stim is None:
        if nx == 2:
            d_stim = np.pi / 2
        else:
            d_stim = 2 * np.pi / nx

    true_M = np.zeros((len(task_list), len(epoch_list), len(epoch_list)))
    for itask, task_type in enumerate(task_list):
        true_M[itask] = np.eye(len(epoch_list))
        epochs = task_dict[task_type].split('->')
        for iepoch, epoch_type in enumerate(epochs[:-1]):
            iz = epoch_list.index(epoch_type)
            iz_nxt = epoch_list.index(epochs[iepoch+1])
            true_M[itask][iz, iz] = p_stay
            true_M[itask][iz, iz_nxt] = 1 - p_stay
    
    true_M += eps
    true_M /= np.sum(true_M, axis=-1, keepdims=True)
    
    nz = len(epoch_list)
    true_W = np.zeros((nx, nz, 6))
    for x in range(nx):
        for z in range(nz):
            true_sy_dict = parse_z_t(epoch_list[z], {'theta_task':x}, 0, 0, d_stim, 1)
            true_W[x, z, :] = np.hstack([true_sy_dict['s'], true_sy_dict['y']])
 
    true_p0_z = np.zeros((nz))
    true_p0_z[np.where(np.array(epoch_list) == 'F/D')[0][0]] = 1
    true_p0_z += eps
    true_p0_z /= np.sum(true_p0_z)
    return true_M, true_W, true_p0_z