import numpy as np

"""
Notations:
Te: duration of epoch
Tt: duration of trial
sigma_n: noise standard deviation

Tasks
PRO: respond to the stimulus, F->S_P->R
PRO_D: respond to the stimulus after delay, F->S_P->D->R
ANTI: respond to the opposite stimulus, F->S_A->R
ANTI_D: respond to the opposite stimulus after delay, F->S_A->D->R
ORDER: respond to the stimulus coming on first, F->S_P->S_B->R
DM: respond to the stronger stimulus, F->S_S->R
DM_D: respond to the stronger stimulus, which may come on first or second, F->S_P->S_AW->R or F->S_AW->S_P->R
"""

epoch_dictionary = {'F': 0, # fixation
            'S_P': 1, # pro stimulus
            'S_A': 2, # anti stimulus
            'S_B': 3, # both stimuli on with equal intensity
            'S_S': 4, # pro stimulus, superimposed with a weak anti stimulus
            'S_AW': 5, # weak anti stimulus
            'D': 6, # delay
            'R': 7} # response


MIN_Te = 50
RESPONSE_Te = 50

def make_stim_input(stim, Te):
    """
    Produce a Te x 2 matrix of noisy inputs for a given stimulus.
    if stim = -1, return zero-mean random noise

    stim: int in [-1, 0, 1]
    duration: number of time steps

    """
    assert stim in [-1, 0, 1]
    if stim == -1:
        return np.zeros((Te, 2))
    else:
        angle = np.pi / 2 * stim
        mean_stim = np.repeat(np.array([[np.cos(angle), np.sin(angle)]]), Te, axis=0)
        return mean_stim


def make_input(stim, fixation, Te, sigma_n, contrast=1, other_stim_contrast=0):
    """
    Generates the input (s) for a given epoch. Assuming a single modality.

    Produces a Te x 3 matrix of noisy inputs for a given stimulus pair and
    fixation cue. The first two columns are the inputs for the first stimulus,
    the second two columns are the inputs for the second stimulus, and the
    last column is the fixation cue.
    """
    assert stim in [-1, 0, 1]

    stim_input = make_stim_input(stim, Te) * contrast

    stim_input += make_stim_input(1 - stim, Te) * other_stim_contrast

    assert fixation in [0, 1]
    fixation_input = np.random.normal(fixation, sigma_n, (Te, 1))

    mean_input = np.hstack([stim_input, fixation_input]).T
    return np.random.normal(mean_input, sigma_n)


def make_bimodal_input(mod1_stim, mod2_stim, fixation, Te, sigma_n, superimpose=False):
    """
    Generates the input (s) for a given epoch.

    Produces a Te x 3 matrix of noisy inputs for a given stimulus pair and
    fixation cue. The first two columns are the inputs for the first stimulus,
    the second two columns are the inputs for the second stimulus, and the
    last column is the fixation cue.
    """
    assert mod1_stim in [-1, 0, 1]
    assert mod2_stim in [-1, 0, 1]

    mod1_stim_input = make_stim_input(mod1_stim, Te)
    mod2_stim_input = make_stim_input(mod2_stim, Te)

    if superimpose:
        # in each modality, superimpose the stimulus with the opposite one
        if mod1_stim != -1:
            mod1_stim_input += make_stim_input(1 - mod1_stim, Te) * 0.5
        elif mod2_stim != -1:
            mod2_stim_input += make_stim_input(1 - mod2_stim, Te) * 0.5

    assert fixation in [0, 1]
    fixation_input = np.random.normal(fixation, sigma_n, (Te, 1))

    mean_input = np.hstack([mod1_stim_input, mod2_stim_input, fixation_input]).T
    return np.random.normal(mean_input, sigma_n)


def make_target_output(response, fixation, Te):
    """
    Generates the target output (y) for a given epoch.
    """
    target_output = make_stim_input(stim=response, Te=Te)
    fixation_output = np.ones((Te, 1)) * fixation
    return np.hstack([target_output, fixation_output]).T


def make_delay_or_fixation_epoch(task_var, Te, sigma_n):
    """
    Generates a delay epoch or fixation epoch (D or F).
    Composed of input (s, Te x 3) and target output (y, Te x 3).
    """

    return {'s':make_input(stim=-1, # no stimulus presentation
                          fixation=1,
                          Te=Te,
                          sigma_n=sigma_n),
            'y':make_target_output(response=-1,  # no stimulus response
                                  fixation=1,
                                  Te=Te)}


def make_response_epoch(task_var, Te, sigma_n):
    """
    Generates a response epoch (R).
    Composed of input (s, Te x 3) and target output (y, Te x 3).
    """

    return {'s':make_input(stim=-1, # no stimulus presentation
                          fixation=0,
                          Te=Te,
                          sigma_n=sigma_n),
            'y':make_target_output(response=task_var,  #respond to the task variable
                                  fixation=0,
                                  Te=Te)}


def make_pro_epoch(task_var, Te, sigma_n):
    """
    Generates a pro-stimulus epoch (S_P).
    Composed of input (s, Te x 3) and target output (y, Te x 3).
    """

    # randomly choose a modality to present the stimulus

    return {'s':make_input(task_var,
                          fixation=1, Te=Te,
                          sigma_n=sigma_n),
            'y':make_target_output(response=-1,
                                  fixation=1,
                                  Te=Te)}


def make_anti_epoch(task_var, Te, sigma_n):
    """
    Generates a anti-stimulus epoch (S_A).
    Composed of input (s, Te x 3) and target output (y, Te x 3).
    """
    
    return make_pro_epoch((task_var + 1) % 2, Te, sigma_n)


def make_superimposed_epoch(task_var, Te, sigma_n):
    """
    Generates a superimposed stimulus epoch (S_S).
    Composed of input (s, Te x 3) and target output (y, Te x 3).
    """

    return {'s':make_input(stim=task_var,
                          fixation=1, Te=Te,
                          sigma_n=sigma_n, other_stim_contrast=0.5),
            'y':make_target_output(response=-1,
                                  fixation=1,
                                  Te=Te)}


def make_both_epoch(task_var, Te, sigma_n):
    """
    Generates a both-stimuli-on epoch (S_B).
    Composed of input (s, Te x 3) and target output (y, Te x 3).
    """

    return {'s':make_input(stim=task_var,
                          fixation=1, Te=Te,
                          sigma_n=sigma_n, other_stim_contrast=1),
            'y':make_target_output(response=-1,
                                  fixation=1,
                                  Te=Te)}


def compose_trial(string_of_epochs:str, task_var, sigma_n, transition_probs=None):
    """
    Generates a trial composed of a sequence of epochs.

    string_of_epochs: a string of epoch symbols separated by '->'
    task_var: the task variable (0 or 1)
    sigma_n: noise standard deviation
    transition_probs (default to None): a list of transition probabilities
    between epochs. If None, p_out for all epochs is 0.01.

    Returns:
    trial: a dictionary containing the input (s) and target output (y) of the trial
    boundaries: a list of boundaries (in units of timesteps) between epochs
    string_of_epochs: the string of epochs used to generate the trial
    """
    epoch_symbols = string_of_epochs.split('->')

    if transition_probs is not None:
        assert len(transition_probs) == len(epoch_symbols) - 1
        
    else:
        transition_probs = np.ones(len(epoch_symbols) - 1) * 0.01
    
    
    all_Te = [np.max([np.random.geometric(prob), MIN_Te]) for prob in transition_probs]
    
    all_Te.append(RESPONSE_Te)  # add a response epoch at the end

    epochs = []
    for Te, epoch_symbol in zip(all_Te, epoch_symbols):
        if epoch_symbol == 'F':
            epochs.append(
                make_delay_or_fixation_epoch(task_var, Te=Te, sigma_n=sigma_n))

        elif epoch_symbol == 'S_P':
            epochs.append(
                make_pro_epoch(task_var, Te=Te, sigma_n=sigma_n))

        elif epoch_symbol == 'S_A':
            epochs.append(
                make_anti_epoch(task_var, Te=Te, sigma_n=sigma_n))

        elif epoch_symbol == 'S_B':
            epochs.append(
                make_both_epoch(task_var, Te=Te, sigma_n=sigma_n))

        elif epoch_symbol == 'S_S':
            epochs.append(
                make_superimposed_epoch(task_var, Te=Te, sigma_n=sigma_n))

        elif epoch_symbol == 'D':
            epochs.append(
                make_delay_or_fixation_epoch(task_var, Te=Te, sigma_n=sigma_n))

        elif epoch_symbol == 'R':
            epochs.append(
                make_response_epoch(task_var, Te=Te, sigma_n=sigma_n))
        else:
            raise ValueError(f"Invalid epoch symbol: {epoch_symbol}")
    
    boundaries = np.cumsum(all_Te)
    return (merge_epochs(epochs), boundaries, string_of_epochs)


def make_go_trial(task_var, sigma_n, anti=False, delay=False):
    """
    Generates a pro/anti go trial with or without delay (PRO/ANTI/PRO_D/ANTI_D).
    Composed of input (s, Tt x 3) and target output (y, Tt x 3).
    """
    if anti is True and delay is True:
        string_of_epochs = 'F->S_A->D->R'
    elif anti is True and delay is False:
        string_of_epochs = 'F->S_A->R'
    elif anti is False and delay is True:
        string_of_epochs = 'F->S_P->D->R'
    else:
        string_of_epochs = 'F->S_P->R'
    
    return compose_trial(string_of_epochs, task_var, sigma_n)


def make_order_trial(task_var, sigma_n, order=1):
    """
    Generates an order trial (ORDER1/ORDER2).
    Composed of input (s, Tt x 3) and target output (y, Tt x 3).
    """

    if order == 1:
        string_of_epochs = 'F->S_P->S_D->R'
    elif order == 2:
        string_of_epochs = 'F->S_D->S_P->R'
    
    return compose_trial(string_of_epochs, task_var, sigma_n)


def make_decision_making_trial(task_var, sigma_n):
    return compose_trial('F->S_S->R', task_var, sigma_n)

def merge_epochs(epochs):
    return {'s':np.hstack([epoch['s'] for epoch in epochs]),
            'y':np.hstack([epoch['y'] for epoch in epochs])}