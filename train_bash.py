import os
import numpy as np
from train_config import save_config, vary_config


def run_main(save_name, num_cpu=1, num_gpu=1, cluster=True):
    if cluster:
        with open(save_name + '.sh', 'w') as f:
            f.write('#!/bin/bash\n'
                    + '\n'
                    + '#SBATCH --nodes=1\n'
                    + f'#SBATCH --cpus-per-task={num_cpu}\n'
                    + f'#SBATCH --gres=gpu:{num_gpu}\n'
                    # + '#SBATCH --nodelist=ax21\n'
                    + f'#SBATCH --output={save_name}.out\n'
                    + '\n'
                    + '\n'
                    + f'python3 train.py --save_name={save_name} '
                    + '\n'
                    + 'exit 0;\n')
        test_cmd = f'sbatch --qos=high-priority {save_name}.sh'
        os.system(test_cmd)
    else:
        test_cmd = f'python3 train.py --save_name={save_name}'
        os.system(test_cmd)
    return


def cxtrnn_config():
    config = dict(seed=0, dim_hid=50, alpha=0.5,
                  gating_type=3, rank=None, share_io=True, nonlin='tanh',
                  init_scale=0.1, sig_r=0, weighted=False,
                  optim='Adam', reset_optim=True, lr=0.01, weight_decay=0,
                  batch_size=256, num_iter=500, n_trials_ts=200,
                  sig_s=0.05, p_stay=0.9, min_Te=5, nx=2, d_stim=np.pi/2,
                  epoch_type=1, fixation_type=1, info_type='z',
                  task_list=['PRO_D', 'PRO_M', 'ANTI_D', 'ANTI_M'],
                  z_list=['F/D', 'S', 'R_P', 'R_M_P', 'R_A', 'R_M_A'],
                  use_task_model=False, task_model_ntrials=np.inf,
                  frz_io_layer=True, verbose=True, ckpt_step=10,
                  save_dir=None, retrain=True, save_ckpt=False,
                  train_fn='train_cxtrnn_sequential')
    config_ranges = {
        #################
        'gating_type': [3],
        'rank': [None],
        # 'gating_type': ['all'],
        # 'rank': [None],
        #################
        'save_ckpt': [True],
        'alpha': [0.1, 0.2],
        'sig_r': [0.05],
        'weighted': [True],
        #################
        'p_stay': [None],
        # 'min_Te': [5],
        #################
        'share_io': [False],
        'frz_io_layer': [False],
        # 'share_io': [True],
        # 'frz_io_layer': [True],
        #################
        # 'epoch_type': [2],
        # 'z_list': [['F', 'D', 'S', 'R_P', 'R_M_P', 'R_A', 'R_M_A']],
        # 'z_list': [['F', 'D', 'S', 'S_S', 'R_P', 'R_M_P', 'R_A', 'R_M_A']],
        #################
        # 'fixation_type': [2],
        # 'nx': [8],
        # 'd_stim': [2 * np.pi / 8],
        # 'dim_hid': [256], #[256, 512],
        # 'nonlin': ['softplus', 'relu'],
        # 'task_list': [['PRO_M']],
        # 'task_list': [['PRO_D', 'PRO_S', 'ANTI_D', 'ANTI_S']],
        # 'task_list': [['PRO_D', 'ANTI_D', 'PRO_M', 'ANTI_M']],
        'task_list': [['PRO_D', 'ANTI_D', 'PRO_M', 'ANTI_M', 'DM'],
                      ['PRO_D', 'PRO_M', 'ANTI_D', 'ANTI_M', 'DM']],
        'z_list': [['F/D', 'S', 'S_S', 'R_P', 'R_M_P', 'R_A', 'R_M_A']],
        # 'num_iter': [30],
        'ckpt_step': [10],
        # 'lr': [1e-4],
        # 'optim': ['SGD'],
        'reset_optim': [False], #[True, False],
        # 'weight_decay': [1e-7],
        ###########################
        # 'use_task_model': [True],
        # 'task_model_ntrials': [512],
        ############################
        'seed': [0],
    }
    configs = vary_config(config, config_ranges,
                          mode=['combinatorial', 'sequential'][0])
    save_names = []
    for config in configs:
        if config['gating_type'] == 'all':
            config['rank'] = len(config['z_list']) * 3
        save_name = 'cxtrnn_seq_gating' + str(config['gating_type'])
        save_name += '_frzio' if config['frz_io_layer'] else ''
        save_name += '_ioZ' if not config['share_io'] else ''
        save_name += ('_a' + str(config['alpha']).replace('.', 'pt')) if config['alpha'] != 0.5 else ''
        save_name += ('_nh' + str(config['dim_hid'])) if config['dim_hid'] != 50 else ''
        save_name += ('_' + str(config['nonlin'])) if config['nonlin'] != 'tanh' else ''
        save_name += ('_sigr' + str(config['sig_r'])).replace('.', 'pt') if config['sig_r'] != 0 else ''
        save_name += '_wloss' if config['weighted'] else ''
        save_name += '_' + ''.join([cur[0] + cur[-1].lower() for cur in config['task_list']])
        # save_name += ('_sigs' + str(config['sig_s'])).replace('.', 'pt') if config['sig_s'] != 0.05 else ''
        config['sig_s'] = np.sqrt(2 / config['alpha']) * 0.01
        save_name += '_sigs'
        save_name += '_dur' if config['p_stay'] is None else ('_minT' + str(config['min_Te']))
        save_name += ('_z' + str(config['epoch_type'])) if config['epoch_type'] != 1 else ''
        save_name += ('_fix' + str(config['fixation_type'])) if config['fixation_type'] != 1 else ''
        save_name += '_nx' + str(config['nx']) + 'dx' + str(int(np.pi/config['d_stim']))
        save_name += ('_nitr' + str(config['num_iter'])) if config['num_iter'] != 500 else ''
        save_name += ('_' + str(config['optim'])) if config['optim'] != 'Adam' else ''
        save_name += '_sameopt' if not config['reset_optim'] else ''
        save_name += ('_lr' + str(config['lr']).replace('.', 'pt')) if config['lr'] != 0.01 else ''
        save_name += ('_wd' + str(config['weight_decay'])) if config['weight_decay'] != 0 else ''
        if config['use_task_model']:
            save_name += '_tskm'
            if np.isfinite(config['task_model_ntrials']):
                save_name += str(config['task_model_ntrials'])
        save_name += '_sd' + str(config['seed'])
        save_names.append(save_name)
    return configs, save_names


def leakyrnn_config():
    config = dict(seed=0, dim_hid=50, dim_s=3, dim_y=3,
                 alpha=0.5, nonlin='tanh', init_scale=0.1, sig_r=0,
                 reset_optim=True, use_proj=False, lr=0.01, weight_decay=0, clip_norm=None,
                 batch_size=256, num_iter=500, n_trials_ts=200, n_trials_vl=200,
                 sig_s=0.05, p_stay=0.9, min_Te=5, nx=2, d_stim=np.pi/2,
                 epoch_type=1, fixation_type=1, info_type='c',
                 task_list=['PRO_D', 'PRO_M', 'ANTI_D', 'ANTI_M'], z_list=None,
                 verbose=True, ckpt_step=10, alpha_cov=0.001,
                 save_dir=None, retrain=True, save_ckpt=False,
                 train_fn='train_leakyrnn_sequential')
    config_ranges = {
        # 'task_list': [['PRO_D', 'ANTI_D', 'PRO_M', 'ANTI_M']],
        # 'dim_hid': [256],
        # 'p_stay': [None],
        # 'num_iter': [2000],
        'ckpt_step': [10],
        'reset_optim': [True, False],
        'use_proj': [True],
        'seed': [0],
    }
    configs = vary_config(config, config_ranges,
                          mode=['combinatorial', 'sequential'][0])
    save_names = []
    for config in configs:
        save_name = 'leakyrnn'
        save_name += '_proj' if config['use_proj'] else ''
        save_name += ('_a' + str(config['alpha']).replace('.', 'pt')) if config['alpha'] != 0.5 else ''
        save_name += ('_nh' + str(config['dim_hid'])) if config['dim_hid'] != 50 else ''
        save_name += ('_' + str(config['nonlin'])) if config['nonlin'] != 'tanh' else ''
        save_name += ('_sigr' + str(config['sig_r'])).replace('.', 'pt') if config['sig_r'] != 0 else ''
        save_name += '_' + ''.join([cur[0] for cur in config['task_list']])
        save_name += '_' + ''.join([cur[-1] for cur in config['task_list']])
        # save_name += ('_sigs' + str(config['sig_s'])).replace('.', 'pt') if config['sig_s'] != 0.05 else ''
        config['sig_s'] = np.sqrt(2 / config['alpha']) * 0.01
        save_name += '_sigs'
        save_name += '_dur' if config['p_stay'] is None else ('_minT' + str(config['min_Te']))
        save_name += ('_z' + str(config['epoch_type'])) if config['epoch_type'] != 1 else ''
        save_name += ('_fix' + str(config['fixation_type'])) if config['fixation_type'] != 1 else ''
        save_name += '_nx' + str(config['nx']) + 'dx' + str(int(np.pi/config['d_stim']))
        save_name += ('_nitr' + str(config['num_iter'])) if config['num_iter'] != 500 else ''
        save_name += '_sameopt' if not config['reset_optim'] else ''
        save_name += ('_lr' + str(config['lr']).replace('.', 'pt')) if config['lr'] != 0.01 else ''
        save_name += ('_wd' + str(config['weight_decay'])) if config['weight_decay'] != 0 else ''
        save_name += ('_clip' + str(config['clip_norm'])) if config['clip_norm'] is not None else ''
        save_name += '_sd' + str(config['seed'])
        save_names.append(save_name)
    return configs, save_names


if __name__ == '__main__':
    configs, save_names = cxtrnn_config()
    for config, save_name in zip(configs, save_names):
        save_config(config, save_name)
        run_main(save_name, num_cpu=1, num_gpu=1)