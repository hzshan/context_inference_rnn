import torch
import pickle
import numpy as np
from copy import deepcopy
from torch.utils.data import DataLoader
from train import train_cxtrnn_sequential, evaluate, task_epoch, get_z_list, TaskDataset, collate_fn, LeakyRNN
from train import AdamWithProj, masked_mse_loss
from train_config import load_config


def train_cxtrnn_sequential_few_shot(save_name):
    config = load_config(save_name)
    config['retrain'] = False
    model, _, _, _ = train_cxtrnn_sequential(**config)
    with open(config['save_dir'] + '/task_model.pkl', 'rb') as f:
        task_model = pickle.load(f)
        _ = pickle.load(f)
    #######################################
    # nc, M, transition_counts, familiar_cx, n_epochs_each_task
    itask_new = len(config['task_list'])
    task_new = 'ANTI_M'
    config['task_list'] += [task_new]
    task_model.nc += 1
    nz = task_model.nz
    M_pad = np.ones((1, nz, nz))
    M_pad /= M_pad.sum(axis=-1, keepdims=True)
    task_model.M = np.concatenate((task_model.M, M_pad), axis=0)
    task_model.transition_counts = np.concatenate((task_model.transition_counts,
                                                   np.zeros((1, nz, nz))), axis=0)
    task_model.familiar_cx = np.concatenate((task_model.familiar_cx,
                                             np.zeros((1, task_model.nx))), axis=0)
    task_model.n_epochs_each_task += [len(set(task_epoch(task_new, epoch_type=config['epoch_type']).split('->')))]
    ########################################
    z_list = get_z_list(config['task_list'], config['epoch_type']) if config['z_list'] is None else config['z_list']
    data_kwargs = dict(dim_s=config['dim_s'], dim_y=config['dim_y'], z_list=z_list, task_list=config['task_list'],
                       sig_s=config['sig_s'], p_stay=config['p_stay'], min_Te=config['min_Te'], nx=config['nx'],
                       d_stim=config['d_stim'], epoch_type=config['epoch_type'], fixation_type=config['fixation_type'],
                       info_type=config['info_type'], min_Te_R=config['min_Te_R'], p_stay_R=config['p_stay_R'])
    ########################################
    n_trials_ts = 200
    ts_data_loaders = []
    for itask, task in enumerate(config['task_list']):
        ts_dataset = TaskDataset(n_trials=n_trials_ts, task=task, **data_kwargs)
        ts_data_loader = DataLoader(dataset=ts_dataset, batch_size=n_trials_ts, collate_fn=collate_fn)
        ts_data_loaders.append(ts_data_loader)
    loss_kwargs = dict(w_fix=config['w_fix'], n_skip=config['n_skip'], reg_act=config['reg_act'])
    strict = True
    ts_loss, ts_perf, ts_err = evaluate(model, ts_data_loaders, loss_kwargs, task_model=task_model, strict=strict)
    ########################################
    ts_loss_arr, ts_perf_arr, ts_err_arr = [ts_loss], [ts_perf], [ts_err]
    n_trials_tr = 512
    tr_dataset = TaskDataset(n_trials=n_trials_tr, task=task_new, **data_kwargs)
    dim_s, dim_y = config['dim_s'], config['dim_y']
    for itrial in range(len(tr_dataset)):
        print(f'trial {itrial}', flush=True)
        trial = tr_dataset.trials[itrial]
        s_ = deepcopy(trial[:, :dim_s].numpy())
        y_ = deepcopy(trial[:, dim_s:(dim_s + dim_y)].numpy())
        if config['fixation_type'] == 2:  # flip back for task model
            s_[:, -1] = 1 - s_[:, -1]
            y_[:, -1] = 1 - y_[:, -1]
        trial_ = (itask_new, None, {'s': s_, 'y': y_})
        task_model.dynamic_initialize_W(trial_)
        task_model.learn_single_trial(trial_)
        ts_loss, ts_perf, ts_err = evaluate(model, ts_data_loaders, loss_kwargs, task_model=task_model, strict=strict)
        ts_loss_arr.append(ts_loss)
        ts_perf_arr.append(ts_perf)
        ts_err_arr.append(ts_err)
    ts_loss_arr = np.array(ts_loss_arr)
    ts_perf_arr = np.array(ts_perf_arr)
    ts_err_arr = np.array(ts_err_arr)
    np.save(config['save_dir'] + '/ts_loss_few_shot.npy', ts_loss_arr)
    np.save(config['save_dir'] + '/ts_perf_few_shot.npy', ts_perf_arr)
    np.save(config['save_dir'] + '/ts_err_few_shot.npy', ts_err_arr)
    with open(config['save_dir'] + '/task_model_few_shot.pkl', 'wb') as f:
        pickle.dump(task_model, f)
    return


def train_leakyrnn_sequential_few_shot(save_name):
    INIT = False
    if save_name[:4] == 'INIT':
        INIT = True
        save_name = save_name[4:]

    config = load_config(save_name)
    config['retrain'] = False
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = LeakyRNN(dim_i=(config['dim_s'] + len(config['task_list'])), dim_y=config['dim_y'],
                     dim_hid=config['dim_hid'], alpha=config['alpha'], nonlin=config['nonlin'],
                     sig_r=config['sig_r']).to(device)
    if INIT:
        use_proj = False
        proj_mtrx_dict = dict(cov_1=0, cov_2=0, cov_3=0, cov_4=0, proj_1=None, proj_2=None, proj_3=None, proj_4=None)
    else:
        use_proj = True
        save_sfx = 'after_itask2'
        save_dict = torch.load(config['save_dir'] + f'/model_{save_sfx}.pth', map_location=torch.device(device))
        model.load_state_dict(save_dict)
        proj_mtrx_dict = torch.load(config['save_dir'] + f'/proj_mtrx_dict_{save_sfx}.pt')
    ########################################
    data_kwargs = dict(dim_s=config['dim_s'], dim_y=config['dim_y'], z_list=None, task_list=config['task_list'],
                       sig_s=config['sig_s'], p_stay=config['p_stay'], min_Te=config['min_Te'], nx=config['nx'],
                       d_stim=config['d_stim'], epoch_type=config['epoch_type'], fixation_type=config['fixation_type'],
                       info_type=config['info_type'], min_Te_R=config['min_Te_R'], p_stay_R=config['p_stay_R'])
    ########################################
    n_trials_ts = 200
    ts_data_loaders = []
    for itask, task in enumerate(config['task_list']):
        ts_dataset = TaskDataset(n_trials=n_trials_ts, task=task, **data_kwargs)
        ts_data_loader = DataLoader(dataset=ts_dataset, batch_size=n_trials_ts, collate_fn=collate_fn)
        ts_data_loaders.append(ts_data_loader)
    loss_kwargs = dict(w_fix=config['w_fix'], n_skip=config['n_skip'], reg_act=config.get('reg_act', 0))
    strict = True
    ts_loss, ts_perf, _ = evaluate(model, ts_data_loaders, loss_kwargs, strict=strict)
    ########################################
    ts_loss_arr, ts_perf_arr = [ts_loss], [ts_perf]
    task_new = config['task_list'][-1]
    n_trials_tr = 512
    batch_size = 1
    optimizer = AdamWithProj(model.named_parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    tr_dataset = TaskDataset(n_trials=n_trials_tr, task=task_new, **data_kwargs)
    tr_data_loader = DataLoader(dataset=tr_dataset, batch_size=batch_size, collate_fn=collate_fn)
    model.train()
    dim_s, dim_y = config['dim_s'], config['dim_y']
    for ib, cur_batch in enumerate(tr_data_loader):
        print(f'batch {ib}', flush=True)
        syz, lengths = cur_batch
        syz = syz.to(device)
        s, y, z = syz[..., :dim_s], syz[..., dim_s:(dim_s + dim_y)], syz[..., (dim_s + dim_y):]
        out, states = model(s, z)
        loss = masked_mse_loss(out, model.nonlin(states), y, lengths, **loss_kwargs)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step(use_proj=use_proj, proj_mtrx_dict=proj_mtrx_dict)
        ts_loss, ts_perf, _ = evaluate(model, ts_data_loaders, loss_kwargs, strict=strict)
        ts_loss_arr.append(ts_loss)
        ts_perf_arr.append(ts_perf)
    ts_loss_arr = np.array(ts_loss_arr)
    ts_perf_arr = np.array(ts_perf_arr)
    save_sfx = '_INIT' if INIT else ''
    np.save(config['save_dir'] + f'/ts_loss_few_shot{save_sfx}.npy', ts_loss_arr)
    np.save(config['save_dir'] + f'/ts_perf_few_shot{save_sfx}.npy', ts_perf_arr)
    return


if __name__ == '__main__':
    import argparse
    import platform
    if platform.system() == 'Windows':
        save_name = 'cxtrnn_seq_v2_gating3_ioZ_relu_PsPmAs_tskm512_sd0'
    else:
        parser = argparse.ArgumentParser()
        parser.add_argument("--save_name", help="save_name", type=str, default="")
        args = parser.parse_args()
        save_name = args.save_name

    if 'cxtrnn' in save_name:
        train_cxtrnn_sequential_few_shot(save_name)
    else:
        train_leakyrnn_sequential_few_shot(save_name)

