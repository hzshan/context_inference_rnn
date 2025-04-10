import os
import torch
import torch.nn as nn
import random
from copy import deepcopy
import numpy as np
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from task_generation import compose_trial
import task_model
import task_generation
import torch.nn.functional as F
import matplotlib.pyplot as plt


def task_epoch(task, epoch_type=1):
    if epoch_type == 1:
        task_dict = {
            'PRO_D': 'F/D->S->R_P',
            'PRO_S': 'F/D->S->R_M_P',
            'PRO_M': 'F/D->S->F/D->R_M_P',
            'PRO_R': 'F/D->R_P',
            'ANTI_D': 'F/D->S->R_A',
            'ANTI_S': 'F/D->S->R_M_A',
            'ANTI_M': 'F/D->S->F/D->R_M_A',
            'ANTI_R': 'F/D->R_A',
            'ORDER': 'F/D->S->S_B->R_M_P',
            'DM': 'F/D->S_S->R_M_P'
        }
    elif epoch_type == 2:
        task_dict = {
            'PRO_D': 'F->S->R_P',
            'PRO_S': 'F->S->R_M_P',
            'PRO_M': 'F->S->D->R_M_P',
            'PRO_R': 'F->R_P',
            'ANTI_D': 'F->S->R_A',
            'ANTI_S': 'F->S->R_M_A',
            'ANTI_M': 'F->S->D->R_M_A',
            'ANTI_R': 'F->R_A',
            'ORDER': 'F->S->S_B->R_M_P',
            'DM': 'F->S_S->R_M_P'
        }
    else:
        raise NotImplementedError
    return task_dict[task]


class CXTRNN(nn.Module):
    def __init__(self, dim_s=3, dim_y=3, dim_z=6, rank=6, dim_hid=50, alpha=0.5,
                 gating_type='nl', share_io=True, nonlin='tanh',
                 init_scale=0.1, sig_r=0, **kwargs):
        super(CXTRNN, self).__init__()
        self.U = nn.Parameter(torch.randn(dim_hid, rank) * init_scale)
        self.V = nn.Parameter(torch.randn(dim_hid, rank) * init_scale)
        self.rank = rank
        self.dim_z = dim_z
        num_io = 1 if share_io else dim_z
        self.W_in = nn.Parameter(torch.randn(num_io, dim_hid, dim_s) * init_scale)
        self.b_in = nn.Parameter(torch.randn(num_io, dim_hid) * init_scale)
        self.W_out = nn.Parameter(torch.randn(num_io, dim_y, dim_hid) * init_scale)
        self.b_out = nn.Parameter(torch.randn(num_io, dim_y) * init_scale)
        self.activation = torch.tanh
        if gating_type in ['l', 'nl']:
            self.nm_layer = nn.Linear(dim_z, rank)
        self.alpha = alpha
        self.dim_hid = dim_hid
        self.gating_type = gating_type
        self.nonlin = getattr(F, nonlin) if nonlin != 'linear' else (lambda a: a)
        self.sig_r = sig_r

    def input_layer(self, s_t, z_t):
        # s_t: batch x dim, z_t: batch x num_z
        W_in = self.W_in.expand(self.dim_z, -1, -1)
        b_in = self.b_in.expand(self.dim_z, -1)
        out = torch.einsum('zhs,bs->bzh', W_in, s_t)
        out = out + b_in
        out = torch.einsum('bz,bzh->bh', z_t, out)
        return out

    def output_layer(self, h_t, z_t):
        # h_t: batch x dim, z_t: batch x num_z
        W_out = self.W_out.expand(self.dim_z, -1, -1)
        b_out = self.b_out.expand(self.dim_z, -1)
        out = torch.einsum('zyh,bh->bzy', W_out, h_t)
        out = out + b_out
        out = torch.einsum('bz,bzy->by', z_t, out)
        return out

    def gating(self, z_t):
        if self.gating_type == 'o':
            return z_t
        elif isinstance(self.gating_type, int):
            return z_t.unsqueeze(-1).expand(-1, -1, self.gating_type).reshape(z_t.shape[0], -1)
        elif self.gating_type == 'l':
            return self.nm_layer(z_t)
        elif self.gating_type == 'nl':
            return torch.sigmoid(self.nm_layer(z_t))
        elif self.gating_type == 'all':
            return torch.ones(z_t.shape[0], self.rank).to(z_t.device)
        else:
            raise NotImplementedError

    def step(self, s_t, z_t, state):
        tmp = torch.einsum('br,hr,kr,bk->bh', self.gating(z_t), self.U, self.V, self.nonlin(state))
        tmp = tmp + np.sqrt(2 / self.alpha) * self.sig_r * torch.randn(*tmp.shape).to(tmp.device)
        x = (1 - self.alpha) * state + self.alpha * (tmp + self.input_layer(s_t, z_t))
        y = self.output_layer(self.nonlin(x), z_t)
        return y, x

    def forward(self, s, z):
        seq_len, batch_size, _ = s.shape
        x_state = torch.zeros(batch_size, self.dim_hid, device=s.device)  # Initialize x state
        outputs = []
        for t in range(seq_len):
            y, x_state = self.step(s[t], z[t], x_state)
            outputs.append(y)
        outputs = torch.stack(outputs, dim=0)
        return outputs


def masked_mse_loss(predictions, targets, lengths):
    seq_len, batch_size, dim_y = predictions.shape
    mask = torch.arange(seq_len).unsqueeze(1) < lengths.unsqueeze(0)  # (seq_len, batch)
    mask = mask.to(predictions.device)
    mse_per_timestep = (predictions - targets) ** 2
    mse_per_timestep = mse_per_timestep * mask.unsqueeze(-1)
    loss = mse_per_timestep.sum() / (mask.sum().float() * dim_y)
    return loss


class MultiTaskDataset(Dataset):
    def __init__(self, n_trials=64, nx=2, sig_s=0.05, sig_y=0, d_stim=np.pi/2, p_stay=0.9, min_Te=5,
                 task_list=['PRO_D', 'PRO_M', 'ANTI_D', 'ANTI_M'],
                 z_list=['F/D', 'S', 'R_P', 'R_M_P', 'R_A', 'R_M_A'], epoch_type=1, fixation_type=1):
        self.trials = []

        for i_trial in range(n_trials):
            task = np.random.choice(task_list)
            x = np.random.randint(nx)
            epoch_str = task_epoch(task, epoch_type=epoch_type)
            sy, boundaries = compose_trial(epoch_str, {'theta_task': x}, sig_s, sig_y, p_stay, min_Te, d_stim=d_stim)
            all_Te = np.diff([0] + list(boundaries))
            z = [[z_list.index(cur)] * Te for cur, Te in zip(epoch_str.split('->'), all_Te)]
            z = np.array([j for i in z for j in i])[:, None]
            if fixation_type == 2:
                sy['s'][:, -1] = 1 - sy['s'][:, -1]
                sy['y'][:, -1] = 1 - sy['y'][:, -1]
            syz = np.concatenate([sy['s'], sy['y'], z], axis=-1)
            self.trials.append(torch.tensor(syz, dtype=torch.float32))

    def __len__(self):
        return len(self.trials)

    def __getitem__(self, idx):
        return self.trials[idx]


def collate_fn(batch):
    lengths = [len(trial) for trial in batch]
    padded_batch = pad_sequence(batch, batch_first=False, padding_value=0.0)
    return padded_batch, torch.tensor(lengths)


def set_deterministic(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return


def evaluate(model, task_list, ts_data_loaders, z_list, device):
    with torch.no_grad():
        ts_loss = []
        for itask_ts, task_ts in enumerate(task_list):
            cur_ts_loss = []
            for ib, cur_batch in enumerate(ts_data_loaders[itask_ts]):
                syz, lengths = cur_batch
                syz = syz.to(device)
                s, y, z = syz[..., :3], syz[..., 3:6], syz[..., -1]
                z = F.one_hot(z.to(torch.int64), num_classes=len(z_list)).float()
                out = model(s, z)
                loss = masked_mse_loss(out, y, lengths)
                cur_ts_loss.append(loss.item())
            ts_loss.append(np.mean(cur_ts_loss))
    return ts_loss


def train_rnn_sequential(seed=0, dim_hid=50, alpha=0.5,
                         gating_type=3, rank=None, share_io=True, nonlin='tanh', init_scale=0.1, sig_r=0,
                         optim='Adam', lr=0.01, weight_decay=0,
                         batch_size=256, num_iter=500, n_trials_ts=200,
                         sig_s=0.05, p_stay=0.9, min_Te=5, nx=2, d_stim=np.pi/2, epoch_type=1, fixation_type=1,
                         task_list=['PRO_D', 'PRO_M', 'ANTI_D', 'ANTI_M'],
                         z_list=['F/D', 'S', 'R_P', 'R_M_P', 'R_A', 'R_M_A'],
                         use_task_model=False, frz_io_layer=False, verbose=True,
                         save_dir=None, retrain=True, save_ckpt=False, **kwargs):
    set_deterministic(seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    rank = len(z_list) * gating_type if isinstance(gating_type, int) else rank
    model = CXTRNN(dim_z=len(z_list), rank=rank, dim_hid=dim_hid, alpha=alpha,
                   gating_type=gating_type, share_io=share_io, nonlin=nonlin,
                   init_scale=init_scale, sig_r=sig_r).to(device)
    print(sum(p.numel() for p in model.parameters() if p.requires_grad), flush=True)

    if save_dir is not None and os.path.isfile(save_dir + '/model.pth') and not retrain:
        save_dict = torch.load(save_dir + '/model.pth', map_location=torch.device(device))
        model.load_state_dict(save_dict)
        tr_loss_arr = np.load(save_dir + '/tr_loss.npy')
        ts_loss_arr = np.load(save_dir + '/ts_loss.npy')
        return model, tr_loss_arr, ts_loss_arr

    ts_data_loaders = []
    for itask, task in enumerate(task_list):
        multitask_dataset_ts = MultiTaskDataset(n_trials=n_trials_ts, task_list=[task], z_list=z_list,
                                                sig_s=sig_s, p_stay=p_stay, min_Te=min_Te,
                                                nx=nx, d_stim=d_stim, epoch_type=epoch_type,
                                                fixation_type=fixation_type)
        ts_data_loader = DataLoader(dataset=multitask_dataset_ts, batch_size=batch_size, collate_fn=collate_fn)
        ts_data_loaders.append(ts_data_loader)

    ts_loss = evaluate(model, task_list, ts_data_loaders, z_list, device)
    tr_loss_arr = [ts_loss[0]]
    ts_loss_arr = [[ts_loss[itask]] for itask in range(len(task_list))]
    ckpt_list = [deepcopy(model.state_dict())]
    ###########################################
    # if use_task_model:
    #     num_epochs_each_task = [len(np.unique(task_generation.task_dict[task].split('->'))) for task in task_list]
    #     w_fixate = task_generation.make_delay_or_fixation_epoch({}, 1, 0, 0, d_stim=d_stim)
    #     w_fixate = np.concatenate([w_fixate['s'], w_fixate['y']], axis=1)
    #     model = task_model.TaskModel(nc=len(task_list), nz=len(z_list), nx=nx, sigma=sig_s, d=w_fixate.shape[-1],
    #                                  n_epochs_each_task=num_epochs_each_task, w_fixate=w_fixate)
    ###########################################
    for itask, task in enumerate(task_list):
        optimizer = getattr(torch.optim, optim)(model.parameters(), lr=lr, weight_decay=weight_decay)
        for iter in range(num_iter):
            multitask_dataset_tr = MultiTaskDataset(n_trials=batch_size, task_list=[task], z_list=z_list,
                                                    sig_s=sig_s, p_stay=p_stay, min_Te=min_Te,
                                                    nx=nx, d_stim=d_stim, epoch_type=epoch_type,
                                                    fixation_type=fixation_type)
            tr_data_loader = DataLoader(dataset=multitask_dataset_tr, batch_size=batch_size, collate_fn=collate_fn)
            model.train()
            if frz_io_layer:
                model.W_in.requires_grad = False
                model.b_in.requires_grad = False
                model.W_out.requires_grad = False
                model.b_out.requries_grad = False
            for ib, cur_batch in enumerate(tr_data_loader):
                syz, lengths = cur_batch
                syz = syz.to(device)
                s, y, z = syz[..., :3], syz[..., 3:6], syz[..., -1]
                z = F.one_hot(z.to(torch.int64), num_classes=len(z_list)).float()
                out = model(s, z)
                loss = masked_mse_loss(out, y, lengths)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            if (iter + 1) % 10 == 0:
                tr_loss_arr.append(loss.item())
                if verbose:
                    print(f'task {task}, iter {iter + 1}, train loss: {loss.item():.4f}', flush=True)
                # evaluate model on all tasks
                model.eval()
                ts_loss = evaluate(model, task_list, ts_data_loaders, z_list, device)
                for itask_ts, task_ts in enumerate(task_list):
                    ts_loss_arr[itask_ts].append(ts_loss[itask_ts])
                    if verbose:
                        print(f'test loss on task {task_ts}: {ts_loss[itask_ts]:.4f}', flush=True)
                if save_ckpt:
                    ckpt_list.append(deepcopy(model.state_dict()))
    tr_loss_arr = np.array(tr_loss_arr)
    ts_loss_arr = np.array(ts_loss_arr)
    if save_dir is not None:
        torch.save(model.state_dict(), save_dir + '/model.pth')
        torch.save(ckpt_list, save_dir + '/ckpt_list.pth')
        np.save(save_dir + '/tr_loss.npy', tr_loss_arr)
        np.save(save_dir + '/ts_loss.npy', ts_loss_arr)
    return model, tr_loss_arr, ts_loss_arr


def train_rnn(model_kwargs, epochs=15, seed=0, n_trials=25600, batch_size=256, save_dir=None):
    set_deterministic(seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = CXTRNN(**model_kwargs).to(device)
    print(sum(p.numel() for p in model.parameters() if p.requires_grad), flush=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8)
    ##################
    dim_z = 6
    multitask_dataset_tr = MultiTaskDataset(n_trials=n_trials)
    multitask_dataset_ts = MultiTaskDataset(n_trials=int(n_trials / 5))
    tr_data_loader = DataLoader(dataset=multitask_dataset_tr, batch_size=batch_size, collate_fn=collate_fn)
    ts_data_loader = DataLoader(dataset=multitask_dataset_ts, batch_size=batch_size, collate_fn=collate_fn)
    tr_loss_arr = []
    ts_loss_arr = []
    ##################
    for epoch in range(epochs):
        ######################
        model.train()
        avg_tr_loss = []
        for ib, cur_batch in enumerate(tr_data_loader):
            syz, lengths = cur_batch
            syz = syz.to(device)
            s, y, z = syz[..., :3], syz[..., 3:6], syz[..., -1]
            z = F.one_hot(z.to(torch.int64), num_classes=dim_z).float()
            out = model(s, z)
            loss = masked_mse_loss(out, y, lengths)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            avg_tr_loss.append(loss.item())
            if ib % 10 == 0:
                print(f'epoch {epoch}, batch {ib}, train loss: {loss.item():.4f}', flush=True)
        tr_loss_arr.append(np.mean(avg_tr_loss))
        scheduler.step()
        #######################
        model.eval()
        with torch.no_grad():
            avg_ts_loss = []
            for ib, cur_batch in enumerate(ts_data_loader):
                syz, lengths = cur_batch
                syz = syz.to(device)
                s, y, z = syz[..., :3], syz[..., 3:6], syz[..., -1]
                z = F.one_hot(z.to(torch.int64), num_classes=dim_z).float()
                out = model(s, z)
                loss = masked_mse_loss(out, y, lengths)
                avg_ts_loss.append(loss.item())
                print(f'epoch {epoch} test, batch {ib}, test loss: {loss.item():.4f}', flush=True)
            ts_loss_arr.append(np.mean(avg_ts_loss))
        ########################
    tr_loss_arr = np.array(tr_loss_arr)
    ts_loss_arr = np.array(ts_loss_arr)
    if save_dir is not None:
        torch.save(model.state_dict(), save_dir + '/model.pth')
        np.save(save_dir + '/tr_loss.npy', tr_loss_arr)
        np.save(save_dir + '/ts_loss.npy', ts_loss_arr)
    return model, tr_loss_arr, ts_loss_arr


if __name__ == '__main__':
    import argparse
    import platform
    from train_config import load_config

    if platform.system() == 'Windows':
        save_name = 'cxtrnn_seq_gating3_sigr0pt05_P_M_minT5_nx2dx2_sd2'
        config = load_config(save_name)
        config['retrain'] = False
    else:
        parser = argparse.ArgumentParser()
        parser.add_argument("--save_name", help="save_name", type=str, default="")
        args = parser.parse_args()
        save_name = args.save_name
        config = load_config(save_name)
    model, tr_loss_arr, ts_loss_arr = train_rnn_sequential(**config)
    ####################################################
    fig, axes = plt.subplots(2, 1, figsize=(5, 6), sharex=True, sharey=True)
    axes[0].plot(tr_loss_arr, color='gray')
    for itask in range(len(config['task_list'])):
        axes[1].plot(ts_loss_arr[itask], color=f'C{itask}', label=config['task_list'][itask])
    axes[1].legend()
    axes[0].set_ylabel('training loss')
    axes[1].set_ylabel('test loss')
    if config['save_dir'] is not None:
        fig.savefig(config['save_dir'] + '/loss.png', bbox_inches='tight')
    ####################################################
    dim_z = len(config['z_list'])
    fig, axes = plt.subplots(6, len(config['task_list']), figsize=(2 * len(config['task_list']), 6),
                             sharey=True, sharex='col')
    if len(axes.shape) == 1:
        axes = axes[:, None]
    for itask, task in enumerate(config['task_list']):
        epoch_str = task_epoch(task, epoch_type=config['epoch_type'])
        x = np.random.randint(config['nx'])
        sig_y = 0
        sig_s = config['sig_s']
        p_stay = config['p_stay']
        min_Te = config['min_Te']
        d_stim = config['d_stim']
        sy, boundaries = compose_trial(epoch_str, {'theta_task': x}, sig_s, sig_y, p_stay, min_Te, d_stim=d_stim)
        all_Te = np.diff([0] + list(boundaries))
        z = [[config['z_list'].index(cur)] * Te for cur, Te in zip(epoch_str.split('->'), all_Te)]
        z = np.array([j for i in z for j in i])[:, None]
        if config.get('fixation_type', 1) == 2:
            sy['s'][:, -1] = 1 - sy['s'][:, -1]
            sy['y'][:, -1] = 1 - sy['y'][:, -1]
        syz = np.concatenate([sy['s'], sy['y'], z], axis=-1)
        syz = torch.tensor(syz[:, None, :], dtype=torch.float32)
        syz = syz.to(next(model.parameters()).device)
        s, y, z = syz[..., :3], syz[..., 3:6], syz[..., -1]
        z = F.one_hot(z.to(torch.int64), num_classes=dim_z).float()
        with torch.no_grad():
            out = model(s, z)
        ########### plot #########
        axes[0, itask].set_title(task)
        for iax in range(3):
            axes[iax, itask].plot(s[:, :, iax].cpu().numpy())
        for iax in range(3):
            axes[iax + 3, itask].plot(out[:, :, iax].cpu().numpy())
            axes[iax + 3, itask].plot(y[:, :, iax].cpu().numpy())
    if config['save_dir'] is not None:
        fig.savefig(config['save_dir'] + '/example.png', bbox_inches='tight')