import os
from copy import deepcopy
import numpy as np
from train_config import save_config, vary_config


def run_main(save_name, num_cpu=1, num_gpu=1, cluster=True, train_few_shot=False):
    main_file = 'train_few_shot.py' if train_few_shot else 'train.py'
    if cluster:
        with open(save_name + '.sh', 'w') as f:
            f.write('#!/bin/bash\n'
                    + '\n'
                    + '#SBATCH --nodes=1\n'
                    + f'#SBATCH --cpus-per-task={num_cpu}\n'
                    + f'#SBATCH --gres=gpu:{num_gpu}\n'
                    + '#SBATCH --account=abbott\n'
                    # + '#SBATCH --nodelist=ax20\n'
                    + f'#SBATCH --output={save_name}.out\n'
                    + '\n'
                    + '\n'
                    + f'python3 {main_file} --save_name={save_name} '
                    + '\n'
                    + 'exit 0;\n')
        test_cmd = f'sbatch --qos=high-priority {save_name}.sh'
        os.system(test_cmd)
    else:
        test_cmd = f'python3 {main_file} --save_name={save_name}'
        os.system(test_cmd)
    return


task_list = ['PRO_D', 'ANTI_D', 'PRO_M', 'ANTI_M', 'PRO_DM', 'ANTI_DM']
task_pairs = []
for task1 in task_list:
    for task2 in task_list:
        cur_task_list = [task1] if task1 == task2 else [task1, task2]
        task_pairs.append(cur_task_list)


def cxtrnn_config():
    config = dict(seed=0, dim_hid=256, alpha=0.1, dim_s=5, dim_y=3,
                  gating_type=3, rank=None, share_io=False, frz_io_layer=False,
                  nonlin='tanh', init_scale=None,
                  sig_r=0.05, w_fix=0.2, n_skip=0, reg_act=0, strict=False,
                  optim='AdamGated', reset_optim=False, lr=1e-3, weight_decay=1e-5,
                  wd_z_eps=1e-3, lr_z_eps=1e-3, gamma=0.5,
                  batch_size=256, num_iter=1000, n_trials_ts=200,
                  sig_s=0.01, p_stay=0.9, min_Te=5, nx=8, d_stim=2*np.pi/8,
                  epoch_type=1, fixation_type=2, info_type='z',
                  min_Te_R=None, p_stay_R=None,
                  task_list=['PRO_D', 'PRO_M', 'ANTI_D', 'ANTI_M'], mixed_train=False, z_list=None,
                  use_task_model=False, task_model_ntrials=np.inf,
                  verbose=True, ckpt_step=10,
                  save_dir=None, retrain=True, save_ckpt=False,
                  train_fn='train_cxtrnn_sequential')
    config_ranges = {
        #################
        'gating_type': [3],
        'nonlin': ['relu'],  # 'tanh',
        ######################
        # 'task_list': task_pairs,
        # 'task_list': [['ANTI_M'], ['PRO_M', 'ANTI_M'], ['ANTI_D', 'ANTI_M'], ['ANTI_DM', 'ANTI_M']],
        'task_list': [
            # ['PRO_S', 'ANTI_S', 'PRO_M', 'ANTI_M'],
            # ['PRO_S', 'PRO_M', 'ANTI_S', 'ANTI_M'],
            ['PRO_D', 'PRO_M', 'ANTI_D', 'ANTI_M', 'PRO_DM', 'ANTI_DM'],
            # ['PRO_D', 'ANTI_D', 'PRO_M', 'ANTI_M', 'PRO_DM', 'ANTI_DM'],
            ['PRO_M', 'PRO_D', 'ANTI_M', 'ANTI_D', 'PRO_DM', 'ANTI_DM'],
            ['PRO_DM', 'ANTI_DM', 'PRO_D', 'ANTI_D', 'PRO_M', 'ANTI_M'],
                    ],
        # 'mixed_train': [True],
        # 'epoch_type': [2],
        ##########################
        'strict': [True],
        'save_ckpt': [False],
        # 'ckpt_step': [1],
        # 'num_iter': [10],
        ###########################
        'use_task_model': [True],
        'task_model_ntrials': [512],
        ############################
        'seed': [0, 1, 2, 3, 4],
    }
    configs = vary_config(config, config_ranges,
                          mode=['combinatorial', 'sequential'][0])
    save_names = []
    for config in configs:
        config['sig_s'] = np.sqrt(2 / config['alpha']) * config['sig_s']
        save_name = 'cxtrnn_seq_v2_gating' + str(config['gating_type'])
        save_name += ('_rk' + str(config['rank'])) if config['rank'] is not None else ''
        save_name += '_frzio' if config['frz_io_layer'] else ''
        save_name += '_ioZ' if not config['share_io'] else ''
        save_name += ('_a' + str(config['alpha'])) if config['alpha'] != 0.1 else ''
        save_name += ('_nh' + str(config['dim_hid'])) if config['dim_hid'] != 256 else ''
        save_name += ('_' + str(config['nonlin'])) if config['nonlin'] != 'tanh' else ''
        save_name += ('_wfix' + str(config['w_fix'])) if config['w_fix'] != 0.2 else ''
        save_name += ('_nskip' + str(config['n_skip'])) if config['n_skip'] != 0 else ''
        save_name += '_' + ''.join([cur.split('_')[0][0] + cur.split('_')[1].lower() for cur in config['task_list']])
        save_name += '_mixed' if config['mixed_train'] else ''
        save_name += ('_pstay' + str(config['p_stay'])) if config['p_stay'] != 0.9 else ''
        save_name += ('_minT' + str(config['min_Te'])) if config['min_Te'] != 5 else ''
        save_name += ('_minTR' + str(config['min_Te_R'])) if config['min_Te_R'] is not None else ''
        save_name += ('_pstayR' + str(config['p_stay_R'])) if config['p_stay_R'] is not None else ''
        save_name += ('_z' + str(config['epoch_type'])) if config['epoch_type'] != 1 else ''
        save_name += ('_fix' + str(config['fixation_type'])) if config['fixation_type'] != 2 else ''
        save_name += ('_nx' + str(config['nx'])) if config['nx'] != 8 else ''
        save_name += ('_nitr' + str(config['num_iter'])) if config['num_iter'] != 1000 else ''
        save_name += ('_' + str(config['optim'])) if config['optim'] != 'AdamGated' else ''
        save_name += ('_wdEps' + str(config['wd_z_eps'])) if config['wd_z_eps'] != 1e-3 else ''
        save_name += ('_lrEps' + str(config['lr_z_eps'])) if config['lr_z_eps'] != 1e-3 else ''
        save_name += ('_gamma' + str(config['gamma'])) if config['gamma'] != 0.5 else ''
        save_name += '_resetopt' if config['reset_optim'] else ''
        save_name += ('_lr' + str(config['lr'])) if config['lr'] != 0.001 else ''
        save_name += ('_wd' + str(config['weight_decay'])) if config['weight_decay'] != 1e-5 else ''
        save_name += ('_reg' + str(config['reg_act'])) if config['reg_act'] != 0 else ''
        if config['use_task_model']:
            save_name += '_tskm'
            if np.isfinite(config['task_model_ntrials']):
                save_name += str(config['task_model_ntrials'])
        save_name += '_sd' + str(config['seed'])
        save_names.append(save_name.replace('.', 'pt'))
    return configs, save_names


def leakyrnn_config():
    config = dict(seed=0, dim_hid=256, dim_s=5, dim_y=3,
                 alpha=0.1, nonlin='tanh', sig_r=0.05, w_fix=0.2, n_skip=0, strict=False,
                 optim='AdamWithProj', reset_optim=False, lr=0.01, weight_decay=1e-5,
                 use_proj=False, use_ewc=False, ewc_lambda=0,
                 batch_size=256, num_iter=1000, n_trials_ts=200, n_trials_vl=200,
                 sig_s=0.01, p_stay=0.9, min_Te=5, nx=8, d_stim=2*np.pi/8,
                 epoch_type=1, fixation_type=2, info_type='c', min_Te_R=None, p_stay_R=None,
                 task_list=['PRO_D', 'PRO_M', 'ANTI_D', 'ANTI_M'], mixed_train=False, z_list=None,
                 verbose=True, ckpt_step=10, alpha_cov=0.001,
                 save_dir=None, retrain=True, save_ckpt=False,
                 train_fn='train_leakyrnn_sequential')
    config_ranges = {
        'nonlin': ['relu'], #'tanh', 'relu'
        'task_list': [
            # ['PRO_D', 'PRO_M', 'ANTI_D', 'ANTI_M', 'PRO_DM', 'ANTI_DM'],
            ['PRO_D', 'ANTI_D', 'PRO_M', 'ANTI_M', 'PRO_DM', 'ANTI_DM'],
            # ['PRO_M', 'PRO_D', 'ANTI_M', 'ANTI_D', 'PRO_DM', 'ANTI_DM'],
            # ['PRO_DM', 'ANTI_DM', 'PRO_D', 'ANTI_D', 'PRO_M', 'ANTI_M'],
                    ],
        # 'mixed_train': [True],
        # 'use_proj': [True],
        ######################
        'use_ewc': [True],
        'ewc_lambda': [5e4],
        ######################
        # 'ckpt_step': [1],
        # 'num_iter': [30],
        'strict': [True],
        'seed': [0, 1, 2, 3, 4],
    }
    configs = vary_config(config, config_ranges,
                          mode=['combinatorial', 'sequential'][0])
    save_names = []
    for config in configs:
        config['sig_s'] = np.sqrt(2 / config['alpha']) * config['sig_s']
        save_name = 'leakyrnn_v1'
        save_name += '_proj' if config['use_proj'] else ''
        save_name += ('_ewc' + str(int(config['ewc_lambda']))) if config['use_ewc'] else ''
        save_name += ('_a' + str(config['alpha'])) if config['alpha'] != 0.1 else ''
        save_name += ('_nh' + str(config['dim_hid'])) if config['dim_hid'] != 256 else ''
        save_name += ('_' + str(config['nonlin'])) if config['nonlin'] != 'tanh' else ''
        save_name += ('_wfix' + str(config['w_fix'])) if config['w_fix'] != 0.2 else ''
        save_name += ('_nskip' + str(config['n_skip'])) if config['n_skip'] != 0 else ''
        save_name += '_' + ''.join([cur.split('_')[0][0] + cur.split('_')[1].lower() for cur in config['task_list']])
        save_name += '_mixed' if config['mixed_train'] else ''
        save_name += ('_pstay' + str(config['p_stay'])) if config['p_stay'] != 0.9 else ''
        save_name += ('_minT' + str(config['min_Te'])) if config['min_Te'] != 5 else ''
        save_name += ('_minTR' + str(config['min_Te_R'])) if config['min_Te_R'] is not None else ''
        save_name += ('_pstayR' + str(config['p_stay_R'])) if config['p_stay_R'] is not None else ''
        save_name += ('_fix' + str(config['fixation_type'])) if config['fixation_type'] != 2 else ''
        save_name += ('_nx' + str(config['nx'])) if config['nx'] != 8 else ''
        save_name += ('_nitr' + str(config['num_iter'])) if config['num_iter'] != 1000 else ''
        save_name += ('_' + str(config['optim'])) if config['optim'] != 'Adam' else ''
        save_name += '_resetopt' if config['reset_optim'] else ''
        save_name += ('_lr' + str(config['lr'])) if config['lr'] != 0.01 else ''
        save_name += ('_wd' + str(config['weight_decay'])) if config['weight_decay'] != 1e-5 else ''
        save_name += '_sd' + str(config['seed'])
        save_names.append(save_name.replace('.', 'pt'))
    return configs, save_names


if __name__ == '__main__':
    configs, save_names = cxtrnn_config()
    for config, save_name in zip(configs, save_names):
        if os.path.isfile(f'./saved_models/{save_name}/ts_perf_strict.npy'):
            continue
        save_config(config, save_name)
        run_main(save_name, num_cpu=1, num_gpu=1)

    # for task_order in ['PsPmAs', 'PsAsPm']:
    #     for seed in range(3):
    #         save_name = f'cxtrnn_seq_v2_gating3_ioZ_relu_{task_order}_tskm512_sd{seed}'
    #         run_main(save_name, num_cpu=1, num_gpu=1, cluster=True, train_few_shot=True)