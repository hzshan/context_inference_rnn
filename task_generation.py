import numpy as np

"""
Notations:
Te: duration of epoch
Tt: duration of trial
sig_s: input noise standard deviation
sig_y: target output noise standard deviation

x: dictionary containing the task variables

Tasks
PRO: respond to the stimulus, F->S_P->R
PRO_D: respond to the stimulus after delay, F->S_P->D->R
ANTI: respond to the opposite stimulus, F->S_A->R
ANTI_D: respond to the opposite stimulus after delay, F->S_A->D->R
ORDER: respond to the stimulus coming on first, F->S_P->S_B->R
DM: respond to the stronger stimulus, F->S_S->R
"""

task_dict = {
    'PRO': 'F->S_P->R',
    'PRO_D': 'F->S_P->D->R',
    'ANTI': 'F->S_A->R',
    'ANTI_D': 'F->S_A->D->R',
    'ORDER1': 'F->S_P->S_B->R',
    'DM': 'F->S_S->R',
}

MIN_Te = 50  # minimum number of time steps in an epoch
RESPONSE_Te = 50  # number of time steps in the response epoch

def make_stim_input(stim: int, Te: int):
    """
    Produce a Te x 2 matrix of noisy inputs for a given stimulus.
    if stim = -1, return zero-mean random noise

    Args:
        stim: int in [-1, 0, 1]
        duration: number of time steps

    Returns:
        mean_stim: Te x 2 matrix of noisy inputs

    """
    assert stim in [-1, 0, 1]

    if stim == -1:
        return np.zeros((Te, 2))
    else:
        angle = np.pi / 2 * stim
        mean_stim = np.repeat(np.array([[np.cos(angle), np.sin(angle)]]), Te, axis=0)
        return mean_stim


def make_input(stim: int, fixation: int, Te: int, sig_s: float,
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
    assert stim in [-1, 0, 1]

    stim_input = make_stim_input(stim, Te) * contrast

    if stim != -1:
        stim_input += make_stim_input(1 - stim, Te) * other_stim_contrast

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


def make_target_output(response: int, fixation: int, Te: int, sig_y: float):
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
    target_output = make_stim_input(stim=response, Te=Te)
    fixation_output = np.ones((Te, 1)) * fixation
    return np.random.normal(
        np.hstack([target_output, fixation_output]), sig_y)


def make_delay_or_fixation_epoch(x: dict, Te: int, sig_s: float, sig_y: float):
    """
    Generates a delay epoch or fixation epoch (D or F).
    Composed of input (s, Te x 3) and target output (y, Te x 3).
    """

    return {'s':make_input(stim=-1, # no stimulus presentation
                          fixation=1,
                          Te=Te,
                          sig_s=sig_s),
            'y':make_target_output(response=-1,  # no stimulus response
                                  fixation=1,
                                  Te=Te,
                                  sig_y=sig_y)}


def make_response_epoch(x: dict, Te: int, sig_s: float, sig_y: float):
    """
    Generates a response epoch (R).
    Composed of input (s, Te x 3) and target output (y, Te x 3).
    """

    return {'s':make_input(stim=-1, # no stimulus presentation
                          fixation=0,
                          Te=Te,
                          sig_s=sig_s),
            'y':make_target_output(response=x['theta_task'],  #respond to the task variable
                                  fixation=0,
                                  Te=Te,
                                  sig_y=sig_y)}


def make_pro_epoch(x: dict, Te: int, sig_s: float, sig_y: float):
    """
    Generates a pro-stimulus epoch (S_P).
    Composed of input (s, Te x 3) and target output (y, Te x 3).
    """

    # randomly choose a modality to present the stimulus

    return {'s':make_input(x['theta_task'],
                          fixation=1, Te=Te,
                          sig_s=sig_s),
            'y':make_target_output(response=-1,
                                  fixation=1,
                                  Te=Te,
                                  sig_y=sig_y)}


def make_anti_epoch(x: dict, Te: int, sig_s: float, sig_y: float):
    """
    Generates a anti-stimulus epoch (S_A).
    Composed of input (s, Te x 3) and target output (y, Te x 3).
    """
    
    return make_pro_epoch((x['theta_task'] + 1) % 2,
                          Te,
                          sig_s=sig_s,
                          sig_y=sig_y)


def make_superimposed_epoch(x: dict, Te: int, sig_s: float, sig_y: float):
    """
    Generates a superimposed stimulus epoch (S_S).
    Composed of input (s, Te x 3) and target output (y, Te x 3).
    """

    return {'s':make_input(stim=x['theta_task'],
                          fixation=1, Te=Te,
                          sig_s=sig_s, other_stim_contrast=0.5),
            'y':make_target_output(response=-1,
                                  fixation=1,
                                  Te=Te,
                                  sig_y=sig_y)}


def make_both_epoch(x: dict, Te: int, sig_s: float, sig_y: float):
    """
    Generates a both-stimuli-on epoch (S_B).
    Composed of input (s, Te x 3) and target output (y, Te x 3).
    """

    return {'s':make_input(stim=x['theta_task'],
                          fixation=1, Te=Te,
                          sig_s=sig_s,
                          other_stim_contrast=1),
            'y':make_target_output(response=-1,
                                  fixation=1,
                                  Te=Te,
                                  sig_y=sig_y)}


def parse_z_t(epoch_symbol: str, x: dict, sig_s: float, sig_y: float, Te: int):
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
    if epoch_symbol == 'F':
        return make_delay_or_fixation_epoch(x, Te=Te, sig_s=sig_s, sig_y=sig_y)

    elif epoch_symbol == 'S_P':
        return make_pro_epoch(x, Te=Te, sig_s=sig_s, sig_y=sig_y)

    elif epoch_symbol == 'S_A':
        return make_anti_epoch(x, Te=Te, sig_s=sig_s, sig_y=sig_y)

    elif epoch_symbol == 'S_B':
        return make_both_epoch(x, Te=Te, sig_s=sig_s, sig_y=sig_y)

    elif epoch_symbol == 'S_S':
        return make_superimposed_epoch(x, Te=Te, sig_s=sig_s, sig_y=sig_y)

    elif epoch_symbol == 'D':
        return make_delay_or_fixation_epoch(x, Te=Te, sig_s=sig_s, sig_y=sig_y)

    elif epoch_symbol == 'R':
        return make_response_epoch(x, Te=Te, sig_s=sig_s, sig_y=sig_y)
    
    else:
        raise ValueError(f'Invalid epoch symbol: {epoch_symbol}')


def compose_trial(seq_of_epochs: str,
                  x: dict,
                  sig_s: float,
                  sig_y: float,
                  transition_probs=None):
    """
    Generates a trial composed of a sequence of epochs.

    seq_of_epochs: a sequence of epoch symbols separated by '->'
    x: the task variable (0 or 1)
    sig_s: noise standard deviation for input
    sig_y: noise standard deviation for target output
    transition_probs (default to None): the i-th element is the probability of
    transitioning to the i+1-th epoch. If None, it is 0.01 for all epochs.

    Returns:
    trial: a dictionary containing the input (s) and target output (y) of the trial
    boundaries: a list of boundaries (in units of timesteps) between epochs
    """
    epoch_symbols = seq_of_epochs.split('->')

    if transition_probs is not None:
        assert len(transition_probs) == len(epoch_symbols) - 1
        
    else:
        transition_probs = np.ones(len(epoch_symbols) - 1) * 0.01
    
    
    all_Te = [np.max([np.random.geometric(prob), MIN_Te]) for prob in transition_probs]
    
    all_Te.append(RESPONSE_Te)  # add a response epoch at the end

    epochs = []
    for Te, epoch_symbol in zip(all_Te, epoch_symbols):
        epochs.append(parse_z_t(epoch_symbol, x, sig_s, sig_y, Te))
    
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

    assert obs.shape == mean.shape

    k = obs.shape[1]

    sq_dist = np.sum((obs - mean) ** 2, axis=1)
    return -np.log(2 * np.pi * sigma**2) * k / 2 - sq_dist / (2 * sigma**2)


def compute_emission_logp(s_t,
                          y_t,
                          z_t: str,
                          x: dict,
                          sigma_s=0.1,
                          sigma_y=0.1):
    """
    Compute log likelihood of observing (s_t, y_t) under a given epoch (z) and 
    task variable (x) for each t in 1,...,s_t.shape[0].

    Args:
        s_t: [s_t.shape[0] x 2] matrix of observed inputs (Te can be 1)
        y_t: [s_t.shape[0] x 2] matrix of observed outputs
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
                           sig_s=0, sig_y=0, Te=num_timesteps)

    logp = multivariate_log_likelihood(
        obs=s_t, mean=epoch_mean['s'], sigma=sigma_s)
    logp += multivariate_log_likelihood(
        obs=y_t, mean=epoch_mean['y'], sigma=sigma_y)
    
    return logp


def get_c_z_transition_matrix(task_dict, p_stay=0.9):
    """
    Generate the transition matrix for the task epochs. Each state is a 
    c, z tuple.

    Args:
        task_dict: a dictionary where the keys are task types (e.g. 'PRO')
        and the corresponding values are sequences of epochs (e.g. 'F->S_P->R')
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

    transition_matrix = np.eye((len(cz_pairs))) * p_stay

    for i, cz_pair in enumerate(cz_pairs):
        epoch_type = cz_pair[1]
        if epoch_type != 'R':
            transition_matrix[i, i+1] = 1 - p_stay
        else:
            transition_matrix[i, i] = 1

    cz_to_ind_hash = {c_z_pair: i for i, c_z_pair in enumerate(cz_pairs)}
    
    return transition_matrix, cz_pairs, cz_to_ind_hash