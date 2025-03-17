import os
import numpy as np
import json
from copy import deepcopy
from collections.abc import Iterable

root_save_dir = './saved_models'


def save_config(config, save_name):
    save_dir = f'{root_save_dir}/{save_name}'
    config['save_name'] = save_name
    config['save_dir'] = save_dir
    os.makedirs(save_dir, exist_ok=True)
    with open(save_dir + '/config.json', 'w') as f:
        json.dump(config, f)
    return


def load_config(save_name):
    save_dir = f'{root_save_dir}/{save_name}'
    if os.path.isfile(save_dir + '/config.json'):
        with open(save_dir + '/config.json', 'r') as f:
            config = json.load(f)
    else:
        raise ValueError('no config file in save_dir {}.'.format(save_dir))
    return config


def vary_config(base_config, config_ranges, mode):
    if mode == 'combinatorial':
        _vary_config = _vary_config_combinatorial
    elif mode == 'sequential':
        _vary_config = _vary_config_sequential
    else:
        raise ValueError('Unknown mode {}'.format(str(mode)))
    configs, config_diffs = _vary_config(base_config, config_ranges)
    return configs


def _vary_config_combinatorial(base_config, config_ranges):
    keys = config_ranges.keys()
    dims = [len(config_ranges[k]) for k in keys]
    n_max = int(np.prod(dims))
    configs, config_diffs = list(), list()
    for i in range(n_max):
        config_diff = dict()
        indices = np.unravel_index(i, shape=dims)
        # Set up new config
        for key, index in zip(keys, indices):
            config_diff[key] = config_ranges[key][index]
        config_diffs.append(config_diff)
        new_config = deepcopy(base_config)
        new_config.update(config_diff)
        configs.append(new_config)
    return configs, config_diffs


def _vary_config_sequential(base_config, config_ranges):
    keys = config_ranges.keys()
    dims = [len(config_ranges[k]) for k in keys]
    n_max = dims[0]
    configs, config_diffs = list(), list()
    for i in range(n_max):
        config_diff = dict()
        for key in keys:
            config_diff[key] = config_ranges[key][i]
        config_diffs.append(config_diff)
        new_config = deepcopy(base_config)
        new_config.update(config_diff)
        configs.append(new_config)
    return configs, config_diffs
