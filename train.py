import os
import torch
import torch.nn as nn
import random
import numpy as np
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from task_generation import *
import torch.nn.functional as F
import matplotlib.pyplot as plt


class CXTRNN(nn.Module):
    def __init__(self, dim_s=3, dim_y=3, dim_z=6, rank=6, dim_hid=50, alpha=0.5,
                 gating_type='nl', nonlin='tanh', init_scale=0.1, **kwargs):
        super(CXTRNN, self).__init__()
        self.U = nn.Parameter(torch.randn(dim_hid, rank) * init_scale)
        self.V = nn.Parameter(torch.randn(dim_hid, rank) * init_scale)
        self.input_layer = nn.Linear(dim_s, dim_hid)
        self.output_layer = nn.Linear(dim_hid, dim_y)
        self.activation = torch.tanh
        self.nm_layer = nn.Linear(dim_z, rank)
        self.alpha = alpha
        self.dim_hid = dim_hid
        self.gating_type = gating_type
        self.nonlin = getattr(torch, nonlin)

    def gating(self, z_t):
        if self.gating_type == 'o':
            return z_t
        elif isinstance(self.gating_type, int):
            return z_t.unsqueeze(-1).expand(-1, -1, self.gating_type).reshape(z_t.shape[0], -1)
        elif self.gating_type == 'l':
            return self.nm_layer(z_t)
        elif self.gating_type == 'nl':
            return torch.sigmoid(self.nm_layer(z_t))
        else:
            raise NotImplementedError

    def step(self, s_t, z_t, state):
        tmp = torch.einsum('br,hr,kr,bk->bh', self.gating(z_t), self.U, self.V, self.nonlin(state))
        x = (1 - self.alpha) * state + self.alpha * (tmp + self.input_layer(s_t))
        y = self.output_layer(self.nonlin(x))
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
                 z_list=['F/D', 'S', 'R_P', 'R_M_P', 'R_A', 'R_M_A']):
        self.trials = []

        for i_trial in range(n_trials):
            task = np.random.choice(task_list)
            x = np.random.randint(nx)
            sy, boundaries = compose_trial(task_dict[task], {'theta_task': x},
                                           sig_s, sig_y, p_stay, min_Te, d_stim=d_stim)
            all_Te = np.diff([0] + list(boundaries))
            z = [[z_list.index(cur)] * Te for cur, Te in zip(task_dict[task].split('->'), all_Te)]
            z = np.array([j for i in z for j in i])[:, None]
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


def train_rnn_sequential(seed=0, dim_hid=50, alpha=0.5,
                         gating_type=3, nonlin='tanh', init_scale=0.1,
                         lr=0.01, weight_decay=0, batch_size=256, num_iter=500, n_trials_ts=200,
                         sig_s=0.05, p_stay=0.9, min_Te=5, nx=2, d_stim=np.pi/2,
                         task_list=['PRO_D', 'PRO_M', 'ANTI_D', 'ANTI_M'],
                         z_list=['F/D', 'S', 'R_P', 'R_M_P', 'R_A', 'R_M_A'],
                         frz_io_layer=False, verbose=True, save_dir=None, retrain=True, **kwargs):
    set_deterministic(seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    rank = len(z_list) * gating_type if isinstance(gating_type, int) else len(z_list)
    model = CXTRNN(dim_z=len(z_list), rank=rank, dim_hid=dim_hid, alpha=alpha,
                   gating_type=gating_type, nonlin=nonlin, init_scale=init_scale).to(device)
    print(sum(p.numel() for p in model.parameters() if p.requires_grad), flush=True)

    if save_dir is not None and os.path.isfile(save_dir + '/model.pth') and not retrain:
        save_dict = torch.load(save_dir + '/model.pth', map_location=torch.device(device))
        model.load_state_dict(save_dict)
        tr_loss_arr = np.load(save_dir + '/tr_loss.npy')
        ts_loss_arr = np.load(save_dir + '/ts_loss.npy')
        U_arr = np.load(save_dir + '/U_arr.npy')
        V_arr = np.load(save_dir + '/V_arr.npy')
        return model, tr_loss_arr, ts_loss_arr, U_arr, V_arr

    ts_data_loaders = []
    for itask, task in enumerate(task_list):
        multitask_dataset_ts = MultiTaskDataset(n_trials=n_trials_ts, task_list=[task], z_list=z_list,
                                                sig_s=sig_s, p_stay=p_stay, min_Te=min_Te,
                                                nx=nx, d_stim=d_stim)
        ts_data_loader = DataLoader(dataset=multitask_dataset_ts, batch_size=batch_size, collate_fn=collate_fn)
        ts_data_loaders.append(ts_data_loader)

    tr_loss_arr = []
    ts_loss_arr = [[] for _ in task_list]
    U_arr = []
    V_arr = []
    for itask, task in enumerate(task_list):
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        for iter in range(num_iter):
            multitask_dataset_tr = MultiTaskDataset(n_trials=batch_size, task_list=[task], z_list=z_list,
                                                    sig_s=sig_s, p_stay=p_stay, min_Te=min_Te,
                                                    nx=nx, d_stim=d_stim)
            tr_data_loader = DataLoader(dataset=multitask_dataset_tr, batch_size=batch_size, collate_fn=collate_fn)
            model.train()
            if frz_io_layer:
                for p in model.input_layer.parameters():
                    p.requires_grad = False
                for p in model.output_layer.parameters():
                    p.requires_grad = False
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
            if iter % 10 == 0:
                tr_loss_arr.append(loss.item())
                if verbose:
                    print(f'task {task}, iter {iter}, train loss: {loss.item():.4f}', flush=True)
                # evaluate model on all tasks
                model.eval()
                with torch.no_grad():
                    for itask_ts, task_ts in enumerate(task_list):
                        avg_ts_loss = []
                        for ib, cur_batch in enumerate(ts_data_loaders[itask_ts]):
                            syz, lengths = cur_batch
                            syz = syz.to(device)
                            s, y, z = syz[..., :3], syz[..., 3:6], syz[..., -1]
                            z = F.one_hot(z.to(torch.int64), num_classes=len(z_list)).float()
                            out = model(s, z)
                            loss = masked_mse_loss(out, y, lengths)
                            avg_ts_loss.append(loss.item())
                        avg_ts_loss = np.mean(avg_ts_loss)
                        ts_loss_arr[itask_ts].append(avg_ts_loss)
                        if verbose:
                            print(f'test loss on task {task_ts}: {avg_ts_loss:.4f}', flush=True)
                U_arr.append(model.U.detach().cpu().numpy())
                V_arr.append(model.V.detach().cpu().numpy())
    tr_loss_arr = np.array(tr_loss_arr)
    ts_loss_arr = np.array(ts_loss_arr)
    U_arr = np.array(U_arr)
    V_arr = np.array(V_arr)
    if save_dir is not None:
        torch.save(model.state_dict(), save_dir + '/model.pth')
        np.save(save_dir + '/tr_loss.npy', tr_loss_arr)
        np.save(save_dir + '/ts_loss.npy', ts_loss_arr)
        np.save(save_dir + '/U_arr.npy', U_arr)
        np.save(save_dir + '/V_arr.npy', V_arr)
    return model, tr_loss_arr, ts_loss_arr, U_arr, V_arr


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
        save_name = 'cxtrnn_seq_gating3_frzio_PPAA_DMDM_minT20_sd2'
        config = load_config(save_name)
        config['retrain'] = False
    else:
        parser = argparse.ArgumentParser()
        parser.add_argument("--save_name", help="save_name", type=str, default="")
        args = parser.parse_args()
        save_name = args.save_name
        config = load_config(save_name)
    model, tr_loss_arr, ts_loss_arr, U_arr, V_arr = train_rnn_sequential(**config)
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
    fig, axes = plt.subplots(dim_z, 1, figsize=(3, 1.5*dim_z), sharex=True)
    for iz in range(dim_z):
        z_t = torch.zeros(dim_z, device=next(model.parameters()).device)
        z_t[iz] = 1
        gating = model.gating(z_t.unsqueeze(0)).detach().cpu().numpy()[0]
        plot_u = (U_arr * np.sqrt(gating)).mean((-1, -2))
        plot_v = (V_arr * np.sqrt(gating)).mean((-1, -2))
        axes[iz].plot(plot_u, label='mean(Uz)')
        axes[iz].plot(plot_v, label='mean(Vz)')
        axes[iz].set_ylabel(config['z_list'][iz])
        axes[iz].legend(loc=(1.05, 0.25))
        ymax = max(plot_u.max(), plot_v.max())
        ymin = min(plot_u.min(), plot_v.min())
        for x in range(len(config['task_list']) + 1):
            axes[iz].plot([x * config['num_iter'] / 10] * 2, [ymin, ymax], alpha=0.2, color='gray')
    if config['save_dir'] is not None:
        fig.savefig(config['save_dir'] + '/weight.png', bbox_inches='tight')
    ######################################################
    fig, axes = plt.subplots(6, len(config['task_list']), figsize=(2 * len(config['task_list']), 6),
                             sharey=True, sharex='col')
    for itask, task in enumerate(config['task_list']):
        epoch_str = task_dict[task]
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