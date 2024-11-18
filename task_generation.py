import numpy as np

def get_stim_input(stim, noise_std, duration=10):
    """
    Produce a T x 2 matrix of noisy inputs for a given stimulus.
    if stim = -1, return zero-mean random noise

    stim: int in [-1, 0, 1]
    noise_std: standard deviation of the noise
    duration: number of time steps

    """
    assert stim in [-1, 0, 1]
    if stim == -1:
        return np.random.normal(0, noise_std, (duration, 2))
    else:
        angle = np.pi / 2 * stim
        mean_stim = np.repeat(np.array([[np.cos(angle), np.sin(angle)]]), duration, axis=0)
        return np.random.normal(mean_stim, noise_std)


def get_input(mod1_stim, mod2_stim, fixation, duration, noise_std):
    """
    Produce a T x 5 matrix of noisy inputs for a given stimulus pair and
    fixation cue. The first two columns are the inputs for the first stimulus,
    the second two columns are the inputs for the second stimulus, and the
    last column is the fixation cue.
    """
    mod1_stim_input = get_stim_input(mod1_stim, noise_std, duration)
    mod2_stim_input = get_stim_input(mod2_stim, noise_std, duration)

    assert fixation in [0, 1]
    fixation_input = np.random.normal(fixation, noise_std, (duration, 1))
    return np.hstack([mod1_stim_input, mod2_stim_input, fixation_input]).T


def get_target_output(response, fixation, duration):
    target_output = get_stim_input(response, 0, duration)
    fixation_output = np.ones((duration, 1)) * fixation
    return np.hstack([target_output, fixation_output]).T


def sample_transition(p, tot_time=1000):
    train = np.random.poisson(p, tot_time)
    return np.where(train > 0)[0][0]


def make_delay_or_fixation_epoch(task_var, duration, noise_std):

    return {'s':get_input(mod1_stim=-1, mod2_stim=-1, 
                          fixation=1,
                          duration=duration,
                          noise_std=noise_std),
            'y':get_target_output(response=-1,
                                  fixation=1,
                                  duration=duration)}


def make_response_epoch(task_var, duration, noise_std):

    return {'s':get_input(mod1_stim=-1, mod2_stim=-1, 
                          fixation=0,
                          duration=duration,
                          noise_std=noise_std),
            'y':get_target_output(response=task_var,
                                  fixation=0,
                                  duration=duration)}


def make_pro_epoch(task_var, duration, noise_std):

    if np.random.binomial(1, 0.5):
        mod1_stim = task_var
        mod2_stim = -1
    else:
        mod1_stim = -1
        mod2_stim = task_var

    return {'s':get_input(mod1_stim, mod2_stim,
                          fixation=1, duration=duration,
                          noise_std=noise_std),
            'y':get_target_output(response=-1,
                                  fixation=1,
                                  duration=duration)}


def make_anti_epoch(task_var, duration, noise_std):
    
    return make_pro_epoch((task_var + 1) % 2, duration, noise_std)


def generate_go_trial(task_var, noise_std, anti=False):
    fixate_to_present = 0.01
    present_to_delay = 0.01
    delay_to_response = 0.01
    min_dur = 50

    fixation_dur = np.max([np.random.geometric(fixate_to_present), min_dur])
    present_dur = np.max([np.random.geometric(present_to_delay), min_dur])
    delay_dur = np.max([np.random.geometric(delay_to_response), min_dur])
    response_dur = 100

    if anti:
        task_var = (task_var + 1) % 2
    if np.random.binomial(1, 0.5):
        stim1 = task_var
        stim2 = -1
    else:
        stim1 = -1
        stim2 = task_var

    fixation_epoch = make_delay_or_fixation_epoch(task_var, fixation_dur, noise_std)
    present_epoch = make_pro_epoch(task_var, present_dur, noise_std)
    delay_epoch = make_delay_or_fixation_epoch(task_var, delay_dur, noise_std)
    response_epoch = make_response_epoch(task_var, response_dur, noise_std)

    boundaries = np.cumsum([fixation_dur, present_dur, delay_dur, response_dur])

    return merge_epochs([fixation_epoch, present_epoch, delay_epoch, response_epoch]), boundaries


def merge_epochs(epochs):
    return {'s':np.hstack([epoch['s'] for epoch in epochs]),
            'y':np.hstack([epoch['y'] for epoch in epochs])}