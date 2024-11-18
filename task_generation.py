import numpy as np

"""
Notations:
Te: duration of epoch
Tt: duration of trial
sigma_n: noise standard deviation
"""

def get_stim_input(stim, sigma_n, Te=10):
    """
    Produce a Te x 2 matrix of noisy inputs for a given stimulus.
    if stim = -1, return zero-mean random noise

    stim: int in [-1, 0, 1]
    sigma_n: standard deviation of the noise
    duration: number of time steps

    """
    assert stim in [-1, 0, 1]
    if stim == -1:
        return np.random.normal(0, sigma_n, (Te, 2))
    else:
        angle = np.pi / 2 * stim
        mean_stim = np.repeat(np.array([[np.cos(angle), np.sin(angle)]]), Te, axis=0)
        return np.random.normal(mean_stim, sigma_n)


def get_input(mod1_stim, mod2_stim, fixation, Te, sigma_n):
    """
    Generates the input (s) for a given epoch.

    Produces a Te x 5 matrix of noisy inputs for a given stimulus pair and
    fixation cue. The first two columns are the inputs for the first stimulus,
    the second two columns are the inputs for the second stimulus, and the
    last column is the fixation cue.
    """
    mod1_stim_input = get_stim_input(mod1_stim, sigma_n, Te)
    mod2_stim_input = get_stim_input(mod2_stim, sigma_n, Te)

    assert fixation in [0, 1]
    fixation_input = np.random.normal(fixation, sigma_n, (Te, 1))
    return np.hstack([mod1_stim_input, mod2_stim_input, fixation_input]).T


def get_target_output(response, fixation, Te):
    """
    Generates the target output (y) for a given epoch.
    """
    target_output = get_stim_input(response, 0, Te)
    fixation_output = np.ones((Te, 1)) * fixation
    return np.hstack([target_output, fixation_output]).T


def make_delay_or_fixation_epoch(task_var, Te, sigma_n):
    """
    Generates a delay epoch or fixation epoch.
    Composed of input (s, Te x 5) and target output (y, Te x 3).
    """

    assert task_var in [0, 1]
    return {'s':get_input(mod1_stim=-1, mod2_stim=-1, # no stimulus presentation
                          fixation=1,
                          Te=Te,
                          sigma_n=sigma_n),
            'y':get_target_output(response=-1,  # no stimulus response
                                  fixation=1,
                                  Te=Te)}


def make_response_epoch(task_var, Te, sigma_n):
    """
    Generates a response epoch.
    Composed of input (s, Te x 5) and target output (y, Te x 3).
    """

    assert task_var in [0, 1]
    return {'s':get_input(mod1_stim=-1, mod2_stim=-1, # no stimulus presentation
                          fixation=0,
                          Te=Te,
                          sigma_n=sigma_n),
            'y':get_target_output(response=task_var,  #respond to the task variable
                                  fixation=0,
                                  Te=Te)}


def make_pro_epoch(task_var, Te, sigma_n):
    """
    Generates a pro-stimulus epoch.
    Composed of input (s, Te x 5) and target output (y, Te x 3).
    """

    # randomly choose a modality to present the stimulus
    if np.random.binomial(1, 0.5):
        mod1_stim = task_var
        mod2_stim = -1
    else:
        mod1_stim = -1
        mod2_stim = task_var

    return {'s':get_input(mod1_stim, mod2_stim,
                          fixation=1, Te=Te,
                          sigma_n=sigma_n),
            'y':get_target_output(response=-1,
                                  fixation=1,
                                  Te=Te)}


def make_anti_epoch(task_var, Te, sigma_n):
    """
    Generates a anti-stimulus epoch.
    Composed of input (s, Te x 5) and target output (y, Te x 3).
    """
    
    return make_pro_epoch((task_var + 1) % 2, Te, sigma_n)


def make_go_trial(task_var, sigma_n, anti=False, delay=False):
    """
    Generates a pro go trial with delay.
    Composed of input (s, Tt x 5) and target output (y, Tt x 3).
    """

    # transition probabilities between epochs
    fixate_to_present = 0.01
    present_to_delay = 0.01
    delay_to_response = 0.01
    min_Te = 50  # set a floor for epoch duration


    fixation_Te = np.max([np.random.geometric(fixate_to_present), min_Te])
    present_Te = np.max([np.random.geometric(present_to_delay), min_Te])
    delay_Te = np.max([np.random.geometric(delay_to_response), min_Te])
    response_Te = 100


    fixation_epoch = make_delay_or_fixation_epoch(task_var, fixation_Te, sigma_n)
    if anti:
        present_epoch = make_anti_epoch(task_var, present_Te, sigma_n)
    else:
        present_epoch = make_pro_epoch(task_var, present_Te, sigma_n)
    delay_epoch = make_delay_or_fixation_epoch(task_var, delay_Te, sigma_n)
    response_epoch = make_response_epoch(task_var, response_Te, sigma_n)

    if delay:
        boundaries = np.cumsum([fixation_Te, present_Te, delay_Te, response_Te])
        return (merge_epochs(
            [fixation_epoch, present_epoch, delay_epoch, response_epoch]), 
            boundaries)
    else:
        boundaries = np.cumsum([fixation_Te, present_Te, response_Te])
        return (merge_epochs(
            [fixation_epoch, present_epoch, response_epoch]), 
            boundaries)




def merge_epochs(epochs):
    return {'s':np.hstack([epoch['s'] for epoch in epochs]),
            'y':np.hstack([epoch['y'] for epoch in epochs])}