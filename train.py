import os
import torch
import pickle
import torch.nn as nn
import random
from copy import deepcopy
import numpy as np
import math
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from task_generation import compose_trial, make_delay_or_fixation_epoch
from task_model import TaskModel
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
            'PRO_DM': 'F/D->DM_S->DM_R_P',
            'ANTI_DM': 'F/D->DM_S->DM_R_A',
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
            'PRO_DM': 'F->DM_S->DM_R_P',
            'ANTI_DM': 'F->DM_S->DM_R_A',
        }
    else:
        raise NotImplementedError
    return task_dict[task]


def get_z_list(task_list, epoch_type):
    epoch_list = []
    for task in task_list:
        epoch_str = task_epoch(task, epoch_type=epoch_type)
        epoch_list += epoch_str.split('->')
    z_list = list(dict.fromkeys(epoch_list))
    return z_list


class CXTRNN(nn.Module):
    def __init__(self, dim_s=3, dim_y=3, dim_z=6, rank=6, dim_hid=50, alpha=0.5,
                 gating_type='nl', share_io=True, nonlin='tanh', sig_r=0, init_scale=None,
                 **kwargs):
        super(CXTRNN, self).__init__()
        if init_scale is None:
            init_scale = (1 / dim_hid) ** 0.25
            if isinstance(gating_type, int):
                init_scale /= math.sqrt(gating_type)
        self.U = nn.Parameter(torch.randn(dim_hid, rank) * init_scale)
        self.V = nn.Parameter(torch.randn(dim_hid, rank) * init_scale)
        self.rank = rank
        self.dim_z = dim_z
        num_io = 1 if share_io else dim_z
        self.W_in = nn.Parameter(torch.randn(num_io, dim_hid, dim_s) * math.sqrt(2 / (dim_hid + dim_s)))
        self.b_in = nn.Parameter(torch.zeros(num_io, dim_hid))
        self.W_out = nn.Parameter(torch.randn(num_io, dim_y, dim_hid) * math.sqrt(2 / (dim_hid + dim_y)))
        self.b_out = nn.Parameter(torch.zeros(num_io, dim_y))
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
        if isinstance(self.gating_type, int):
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
        tmp = tmp + math.sqrt(2 / self.alpha) * self.sig_r * torch.randn(*tmp.shape).to(tmp.device)
        state = (1 - self.alpha) * state + self.alpha * (tmp + self.input_layer(s_t, z_t))
        output = self.output_layer(self.nonlin(state), z_t)
        return output, state

    def forward(self, s, z):
        seq_len, batch_size, _ = s.shape
        outputs = []
        states = []
        state = torch.zeros(batch_size, self.dim_hid, device=s.device)
        for t in range(seq_len):
            output, state = self.step(s[t], z[t], state)
            outputs.append(output)
            states.append(state)
        outputs = torch.stack(outputs, dim=0)
        states = torch.stack(states, dim=0)
        return outputs, states


class LeakyRNN(nn.Module):
    def __init__(self, dim_i=3, dim_y=3, dim_hid=50, alpha=0.5, nonlin='tanh', sig_r=0, **kwargs):
        super(LeakyRNN, self).__init__()
        init_scale = math.sqrt(2 / (dim_i + dim_hid + 1 + dim_hid))
        self.Wb_in = nn.Parameter(torch.randn(dim_hid, dim_i + dim_hid + 1) * init_scale)
        self.Wb_out = nn.Parameter(torch.randn(dim_y, dim_hid + 1) * math.sqrt(2 / (dim_hid + 1 + dim_y)))
        self.alpha = alpha
        self.dim_hid = dim_hid
        self.dim_i = dim_i
        self.dim_y = dim_y
        self.nonlin = getattr(F, nonlin) if nonlin != 'linear' else (lambda a: a)
        self.sig_r = sig_r

    def step(self, inp, state):
        ones = torch.ones(inp.shape[0], 1, device=inp.device)
        inp_cat = torch.cat([inp, self.nonlin(state), ones], dim=-1)
        tmp = inp_cat @ self.Wb_in.T
        noise = math.sqrt(2 / self.alpha) * self.sig_r * torch.randn(*tmp.shape).to(tmp.device)
        state = (1 - self.alpha) * state + self.alpha * (tmp + noise)
        hid_cat = torch.cat([self.nonlin(state), ones], dim=-1)
        output = hid_cat @ self.Wb_out.T
        return output, state

    def forward(self, s, c):
        inputs = torch.cat([s, c], dim=-1)
        seq_len, batch_size, _ = inputs.shape
        outputs = []
        states = []
        state = torch.zeros(batch_size, self.dim_hid, device=inputs.device)
        for t in range(seq_len):
            output, state = self.step(inputs[t], state)
            outputs.append(output)
            states.append(state)
        outputs = torch.stack(outputs, dim=0)
        states = torch.stack(states, dim=0)
        return outputs, states


def masked_mse_loss(predictions, states, targets, lengths, w_fix=1, n_skip=0, reg_act=0):
    seq_len, batch_size, dim_y = predictions.shape
    mask = torch.arange(seq_len).unsqueeze(1) < lengths.unsqueeze(0)  # (seq_len, batch)
    mask = mask.to(predictions.device).unsqueeze(-1)
    w = targets[:, :, -1].clone()
    w = (1 - w) if w[0, 0] == 1 else w
    w[w == 0] = w_fix
    grace_idx = w.argmax(dim=0)[None, :] + torch.arange(n_skip)[:, None].to(w.device)
    grace_idx = torch.clamp(grace_idx, max=w.shape[0] - 1)
    trial_idx = torch.arange(w.shape[1])[None, :].expand(n_skip, -1).to(w.device)
    w[grace_idx, trial_idx] = 0
    w_mask = mask * w.unsqueeze(-1)
    loss = torch.sum(torch.square(predictions - targets) * w_mask) / (w_mask.sum() * dim_y)
    loss = loss + reg_act * torch.sum(torch.square(states) * mask) / (mask.sum() * states.shape[-1])
    return loss


@torch.no_grad()
def batch_performance(out, y, lengths, strict=False):
    seq_len, batch_size, dim_y = out.shape
    if strict:
        out_theta = torch.arctan2(out[:, :, 1], out[:, :, 0])
        y_theta = torch.arctan2(y[:, :, 1], y[:, :, 0])
        d_theta = torch.remainder(out_theta - y_theta + math.pi, 2 * math.pi) - math.pi
        response_correct = d_theta.abs() < math.pi / 10
        mask = torch.arange(seq_len).unsqueeze(1) < lengths.unsqueeze(0)  # seq_len, batch_sze
        mask = mask.to(y.device) & (y[:, :, -1] == y[lengths[0] - 1, 0, -1])
        response_correct = torch.all(response_correct | ~mask, dim=0)
        fixation_correct = (out[..., -1] > 0.5) == (y[..., -1] > 0.5)
        mask = torch.arange(seq_len).unsqueeze(1) < lengths.unsqueeze(0)
        fixation_correct = torch.all(fixation_correct | ~mask.to(y.device), dim=0)
        correct = response_correct & fixation_correct
    else:
        out_end = out[lengths - 1, torch.arange(batch_size)]
        out_theta = torch.arctan2(out_end[:, 1], out_end[:, 0])
        y_end = y[lengths - 1, torch.arange(batch_size)]
        y_theta = torch.arctan2(y_end[:, 1], y_end[:, 0])
        d_theta = torch.remainder(out_theta - y_theta + math.pi, 2 * math.pi) - math.pi
        response_correct = d_theta.abs() < math.pi / 10
        correct = response_correct
    return correct


def generate_trial(task, x, z_list, task_list, sig_s, sig_y, p_stay, min_Te, d_stim,
                   epoch_type, fixation_type, info_type='z',
                   epoch_str=None, min_Te_R=None, p_stay_R=None):
    if epoch_str is None:
        epoch_str = task_epoch(task, epoch_type=epoch_type)
    sy, boundaries = compose_trial(epoch_str, {'theta_task': x}, sig_s, sig_y, p_stay, min_Te,
                                   d_stim=d_stim, min_Te_R=min_Te_R, p_stay_R=p_stay_R)
    if fixation_type == 2:
        sy['s'][:, -1] = 1 - sy['s'][:, -1]
        sy['y'][:, -1] = 1 - sy['y'][:, -1]
    trial = [sy['s'], sy['y']]
    if info_type == 'z':
        all_Te = np.diff([0] + list(boundaries))
        z = [[z_list.index(cur)] * Te for cur, Te in zip(epoch_str.split('->'), all_Te)]
        z = np.array([j for i in z for j in i])
        onehot_z = np.zeros((len(z), len(z_list)))
        onehot_z[np.arange(len(z)), z] = 1
        trial.append(onehot_z)
    else:
        c = task_list.index(task)
        onehot_c = np.zeros((len(sy['s']), len(task_list)))
        onehot_c[:, c] = 1
        trial.append(onehot_c)
    trial = np.concatenate(trial, axis=-1)
    return trial


class TaskDataset(Dataset):
    def __init__(self, n_trials=64, dim_s=5, dim_y=3, nx=8, sig_s=0.05, sig_y=0,
                 d_stim=np.pi / 4, p_stay=0.9, min_Te=5,
                 task='PRO_D', task_list=['PRO_D', 'PRO_M', 'ANTI_D', 'ANTI_M'],
                 z_list=['F/D', 'S', 'R_P', 'R_M_P', 'R_A', 'R_M_A'],
                 epoch_type=1, fixation_type=1, info_type='z', min_Te_R=None, p_stay_R=None):
        self.dim_s = dim_s
        self.dim_y = dim_y
        self.fixation_type = fixation_type
        self.trials = []
        self.gdth_c = []
        for i_trial in range(n_trials):
            x = np.random.randint(nx)
            cur_task = np.random.choice(task_list) if task == 'all' else task
            syz = generate_trial(cur_task, x, z_list, task_list, sig_s, sig_y, p_stay, min_Te, d_stim,
                                 epoch_type, fixation_type, info_type=info_type,
                                 min_Te_R=min_Te_R, p_stay_R=p_stay_R)
            self.trials.append(torch.tensor(syz, dtype=torch.float32))
            self.gdth_c.append(task_list.index(cur_task))

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


@torch.no_grad()
def task_model_inference(task_model, c, obs):
    device = obs.device
    Q = torch.tensor(task_model.Q, device=device).float()
    Lambda = torch.tensor(task_model.Lambda, device=device).float()
    Pi = torch.tensor(task_model.Pi, device=device).float()
    sigma = task_model.sigma
    ###############
    nc, nx = Lambda.shape[0], Q.shape[0]
    prior_cx = torch.zeros((nc, nx), device=device)
    prior_cx[c] = 1
    prior_cx /= prior_cx.sum()
    ###############
    obs = obs.to(device)
    nT, nb, k = obs.shape
    obs = obs.permute(1, 0, 2) # nb, nT, k
    ###############
    # nb, nx, nz, nT
    sq_dist = torch.sum(torch.square(Q[None, :, :, None, :k] - obs[:, None, None, :, :]), dim=-1)
    log_likes = - torch.log(torch.tensor(2 * torch.pi * sigma ** 2).float()) * k / 2 - sq_dist / (2 * sigma ** 2)
    ################
    nz = Lambda.shape[1]
    alpha = torch.zeros((nb, nc, nx, nz, nT), device=device)
    alpha[..., 0] = torch.log(Pi) + log_likes[:, None, :, :, 0]
    for it in range(nT - 1):
        m = torch.max(alpha[..., it], dim=-1, keepdim=True).values
        alpha[..., it + 1] = torch.log(torch.einsum('bcxi,cij->bcxj', torch.exp(alpha[..., it] - m),
                                        Lambda)) + m + log_likes[:, None, :, :, it + 1]
    gamma = alpha + torch.log(prior_cx[None, :, :, None, None])
    gamma = gamma - torch.logsumexp(gamma, dim=(1, 2, 3), keepdim=True)
    gamma = torch.exp(gamma)
    inferred_z = gamma.sum(dim=(1, 2))  # nb, nz, nT
    inferred_z = inferred_z.permute(2, 0, 1) # nT, nb, nz
    return inferred_z


@torch.no_grad()
def update_inferred_z(itask, s, y, z, lengths, task_model, use_y=True, fixation_type=1):
    if fixation_type == 2:
        s = s.clone().detach()
        y = y.clone().detach()
        s[..., -1] = 1 - s[..., -1]  # flip back for task model
        y[..., -1] = 1 - y[..., -1]
    obs = torch.cat((s, y), dim=-1) if use_y else s
    inferred_z = task_model_inference(task_model, itask, obs)
    seq_len, batch_size, _ = s.shape
    mask = (torch.arange(seq_len).unsqueeze(1) < lengths.unsqueeze(0)).to(s.device)  # (seq_len, batch)
    inferred_z = inferred_z * mask[..., None]
    max_error = torch.max(torch.abs(inferred_z - z) * mask[..., None]).item()
    return inferred_z, max_error


def check_epoch_occurrence(task, epoch_type, z_list, occurred_z, device):
    if task == 'all':
        gdth_occurred_z = torch.ones(len(z_list)).to(device)
    else:
        gdth_task_epochs = task_epoch(task, epoch_type=epoch_type).split('->')
        gdth_occurred_z = torch.tensor([cur_z in gdth_task_epochs for cur_z in z_list]).float().to(device)
    assert torch.all((occurred_z == gdth_occurred_z).bool())
    return


@torch.no_grad()
def evaluate(model, ts_data_loaders, loss_kwargs, task_model=None, strict=False):
    model.eval()
    device = next(model.parameters()).device
    ts_loss, ts_perf, ts_err = [], [], []
    for itask_ts, ts_data_loader in enumerate(ts_data_loaders):
        cur_ts_loss, cur_ts_perf, cur_ts_err = [], [], []
        dim_s = ts_data_loader.dataset.dim_s
        dim_y = ts_data_loader.dataset.dim_y
        fixation_type = ts_data_loader.dataset.fixation_type
        for ib, cur_batch in enumerate(ts_data_loader):
            syz, lengths = cur_batch
            syz = syz.to(device)
            s, y, z = syz[..., :dim_s], syz[..., dim_s:(dim_s + dim_y)], syz[..., (dim_s + dim_y):]
            ################################
            if task_model is not None:
                z, max_error = update_inferred_z(itask_ts, s, y, z, lengths, task_model,
                                                 use_y=False, fixation_type=fixation_type)
                cur_ts_err.append(max_error)
            else:
                cur_ts_err.append(-1)
            ################################
            out, states = model(s, z)
            loss = masked_mse_loss(out, model.nonlin(states), y, lengths, **loss_kwargs)
            cur_ts_loss.append(loss.item())
            perf = batch_performance(out, y, lengths, strict=strict)
            cur_ts_perf.append(perf.float().mean().item())
        ts_loss.append(np.mean(cur_ts_loss))
        ts_perf.append(np.mean(cur_ts_perf))
        ts_err.append(np.mean(cur_ts_err))
    return ts_loss, ts_perf, ts_err


def get_optimizer(model, optim, lr=0.01, weight_decay=0):
    if optim == 'AdamGated':
        optimizer = AdamGated(model.named_parameters(), lr=lr, weight_decay=weight_decay)
    else:
        optimizer = getattr(torch.optim, optim)(model.parameters(), lr=lr, weight_decay=weight_decay)
    return optimizer


def train_cxtrnn_sequential(seed=0, dim_hid=50, dim_s=5, dim_y=3, alpha=0.5, init_scale=None,
                         gating_type=3, rank=None, share_io=True, nonlin='tanh',
                         sig_r=0, w_fix=1, n_skip=0, reg_act=0, strict=False,
                         optim='Adam', reset_optim=True, lr=0.01, weight_decay=0,
                         wd_z_eps=0, lr_z_eps=0, gamma=1,
                         batch_size=256, num_iter=500, n_trials_ts=200,
                         sig_s=0.05, p_stay=0.9, min_Te=5, nx=2, d_stim=np.pi/2,
                         epoch_type=1, fixation_type=1, info_type='z', min_Te_R=None, p_stay_R=None,
                         task_list=['PRO_D', 'PRO_M', 'ANTI_D', 'ANTI_M'], mixed_train=False, z_list=None,
                         use_task_model=False, task_model_ntrials=np.inf, frz_io_layer=False,
                         verbose=True, ckpt_step=10,
                         save_dir=None, retrain=True, save_ckpt=False, **kwargs):
    set_deterministic(seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    z_list = get_z_list(task_list, epoch_type) if z_list is None else z_list
    rank = len(z_list) * gating_type if isinstance(gating_type, int) else rank
    model = CXTRNN(dim_s=dim_s, dim_y=dim_y, dim_z=len(z_list), rank=rank,
                   dim_hid=dim_hid, alpha=alpha, init_scale=init_scale,
                   gating_type=gating_type, share_io=share_io, nonlin=nonlin, sig_r=sig_r).to(device)

    if save_dir is not None and os.path.isfile(save_dir + '/model.pth') and not retrain:
        save_dict = torch.load(save_dir + '/model.pth', map_location=torch.device(device))
        model.load_state_dict(save_dict)
        tr_loss_arr = np.load(save_dir + '/tr_loss.npy')
        ts_loss_arr = np.load(save_dir + '/ts_loss.npy')
        perf_sfx = '_strict' if strict else ''
        ts_perf_arr = np.load(save_dir + f'/ts_perf{perf_sfx}.npy')
        return model, tr_loss_arr, ts_loss_arr, ts_perf_arr

    print(sum(p.numel() for p in model.parameters() if p.requires_grad), flush=True)
    data_kwargs = dict(dim_s=dim_s, dim_y=dim_y, z_list=z_list, task_list=task_list,
                       sig_s=sig_s, p_stay=p_stay, min_Te=min_Te, nx=nx, d_stim=d_stim,
                       epoch_type=epoch_type, fixation_type=fixation_type,
                       info_type=info_type, min_Te_R=min_Te_R, p_stay_R=p_stay_R)
    ts_data_loaders = []
    for itask, task in enumerate(task_list):
        ts_dataset = TaskDataset(n_trials=n_trials_ts, task=task, **data_kwargs)
        ts_data_loader = DataLoader(dataset=ts_dataset, batch_size=batch_size, collate_fn=collate_fn)
        ts_data_loaders.append(ts_data_loader)
    ###########################################
    task_model = None
    if use_task_model:
        n_epochs_each_task = [len(set(task_epoch(task, epoch_type=epoch_type).split('->'))) for task in task_list]
        q_fixate = make_delay_or_fixation_epoch({}, 1, 0, 0, d_stim=d_stim)
        q_fixate = np.concatenate([q_fixate['s'], q_fixate['y']], axis=1)
        task_model = TaskModel(nc=len(task_list), nz=len(z_list), nx=nx, sigma=sig_s, d=q_fixate.shape[-1],
                               n_epochs_each_task=n_epochs_each_task, q_fixate=q_fixate)
    ###########################################
    loss_kwargs = dict(w_fix=w_fix, n_skip=n_skip, reg_act=reg_act)
    ts_loss, ts_perf, ts_err = evaluate(model, ts_data_loaders, loss_kwargs, task_model=task_model, strict=strict)
    tr_loss_arr = [ts_loss[0]]
    ts_loss_arr = [[ts_loss[itask]] for itask in range(len(task_list))]
    ts_perf_arr = [[ts_perf[itask]] for itask in range(len(task_list))]
    ts_err_arr = [[ts_err[itask]] for itask in range(len(task_list))]
    ckpt_list = [deepcopy(model.state_dict())]
    task_model_ckpt_list = [deepcopy(task_model)]
    ############################################
    optim_kwargs = dict(lr=lr, weight_decay=weight_decay)
    optimizer = get_optimizer(model, optim, **optim_kwargs)
    lr_z = lr * torch.ones(len(z_list)).to(device)
    for itask, task in enumerate(task_list):
        if mixed_train:
            task = 'all'
        if reset_optim:
            optimizer = get_optimizer(model, optim, **optim_kwargs)
        for i_iter in range(num_iter):
            tr_dataset = TaskDataset(n_trials=batch_size, task=task, **data_kwargs)
            #################################################
            if use_task_model:
                if batch_size * (i_iter + 1) <= task_model_ntrials:
                    for itrial in range(len(tr_dataset)):
                        trial = tr_dataset.trials[itrial]
                        s_ = deepcopy(trial[:, :dim_s].numpy())
                        y_ = deepcopy(trial[:, dim_s:(dim_s + dim_y)].numpy())
                        if fixation_type == 2:  # flip back for task model
                            s_[:, -1] = 1 - s_[:, -1]
                            y_[:, -1] = 1 - y_[:, -1]
                        trial_ = (itask, None, {'s': s_, 'y': y_})
                        task_model.incremental_initialize_Q(trial_)
                        task_model.learn_single_trial(trial_)
            #################################################
            tr_data_loader = DataLoader(dataset=tr_dataset, batch_size=batch_size, collate_fn=collate_fn)
            model.train()
            if frz_io_layer:
                model.W_in.requires_grad = False
                model.b_in.requires_grad = False
                model.W_out.requires_grad = False
                model.b_out.requries_grad = False
            for ib, cur_batch in enumerate(tr_data_loader):
                syz, lengths = cur_batch
                syz = syz.to(device)
                s, y, z = syz[..., :dim_s], syz[..., dim_s:(dim_s + dim_y)], syz[..., (dim_s + dim_y):]
                ##########################
                if use_task_model:
                    z, max_error = update_inferred_z(itask, s, y, z, lengths, task_model,
                                                     use_y=True, fixation_type=fixation_type)
                    if verbose and (i_iter in [0, 1, num_iter - 1]):
                        print(f'task {task}, iter {i_iter + 1}, max inference error: {max_error:.4f}', flush=True)
                ##########################
                out, states = model(s, z)
                loss = masked_mse_loss(out, model.nonlin(states), y, lengths, **loss_kwargs)
                optimizer.zero_grad()
                loss.backward()
                if optim == 'AdamGated':
                    occurred_z = (z.sum((0, 1)) / lengths.sum() >= wd_z_eps).float()
                    check_epoch_occurrence(task, epoch_type, z_list, occurred_z, device)
                    optimizer.step(wd_occurred_z=occurred_z, gating_type=gating_type, lr_z=lr_z)
                    if i_iter == num_iter - 1:
                        occurred_z = (z.sum((0, 1)) / lengths.sum() >= lr_z_eps).float()
                        check_epoch_occurrence(task, epoch_type, z_list, occurred_z, device)
                        lr_z = lr_z * gamma ** occurred_z
                else:
                    optimizer.step()
            if (i_iter + 1) % ckpt_step == 0:
                tr_loss_arr.append(loss.item())
                if verbose:
                    print(f'task {task}, iter {i_iter + 1}, train loss: {loss.item():.4f}', flush=True)
                ts_loss, ts_perf, ts_err = evaluate(model, ts_data_loaders, loss_kwargs,
                                                    task_model=task_model, strict=strict)
                for itask_ts, task_ts in enumerate(task_list):
                    ts_loss_arr[itask_ts].append(ts_loss[itask_ts])
                    ts_perf_arr[itask_ts].append(ts_perf[itask_ts])
                    ts_err_arr[itask_ts].append(ts_err[itask_ts])
                    if verbose:
                        print(f'test on task {task_ts}: loss {ts_loss[itask_ts]:.4f}, '
                              f'perf {ts_perf[itask_ts]:.4f}',
                              f'infr err {ts_err[itask_ts]:.4f}', flush=True)
                if save_ckpt:
                    ckpt_list.append(deepcopy(model.state_dict()))
                    task_model_ckpt_list.append(deepcopy(task_model))

    tr_loss_arr = np.array(tr_loss_arr)
    ts_loss_arr = np.array(ts_loss_arr)
    ts_perf_arr = np.array(ts_perf_arr)
    if save_dir is not None:
        torch.save(model.state_dict(), save_dir + '/model.pth')
        torch.save(ckpt_list, save_dir + '/ckpt_list.pth')
        np.save(save_dir + '/tr_loss.npy', tr_loss_arr)
        np.save(save_dir + '/ts_loss.npy', ts_loss_arr)
        perf_sfx = '_strict' if strict else ''
        np.save(save_dir + f'/ts_perf{perf_sfx}.npy', ts_perf_arr)
        if use_task_model:
            np.save(save_dir + '/ts_err.npy', np.array(ts_err_arr))
            with open(save_dir + '/task_model.pkl', 'wb') as f:
                pickle.dump(task_model, f)
                pickle.dump(task_model_ckpt_list, f)
    return model, tr_loss_arr, ts_loss_arr, ts_perf_arr


class AdamGated(torch.optim.Optimizer):
    def __init__(self, named_params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
        named_params = list(named_params)
        params = [p for _, p in named_params]
        self._name = {p: n for n, p in named_params}
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None, wd_occurred_z=None, gating_type=None, lr_z=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr, eps = group["lr"], group["eps"]
            beta1, beta2 = group["betas"]
            wd = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                # ===== weight decay =====
                if wd != 0:
                    if wd_occurred_z is None:
                        grad = grad.add(p, alpha=wd)
                    else:
                        if self._name[p] in ['W_in', 'b_in', 'W_out', 'b_out'] and p.shape[0] == len(wd_occurred_z):
                            grad = grad + wd * p * wd_occurred_z.view(-1, *[1] * (p.ndim - 1))
                        elif self._name[p] in ['U', 'V'] and isinstance(gating_type, int):
                            grad = grad + wd * p * wd_occurred_z[:, None].expand(-1, gating_type).flatten()
                        else:
                            grad = grad.add(p, alpha=wd)
                # ===== state & moments =====
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = torch.zeros((), dtype=torch.int64)
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)
                state["step"] += 1
                t = state["step"].item()
                m, v = state["exp_avg"], state["exp_avg_sq"]
                m.mul_(beta1).add_(grad, alpha=1 - beta1)
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                bias_c1 = 1 - beta1 ** t
                bias_c2 = 1 - beta2 ** t
                m_hat = m / bias_c1
                v_hat = v / bias_c2
                update = - m_hat / (v_hat.sqrt() + eps)
                if lr_z is None:
                    update = lr * update
                else:
                    if self._name[p] in ['W_in', 'b_in', 'W_out', 'b_out'] and p.shape[0] == len(lr_z):
                        update = update * lr_z.view(-1, *[1] * (update.ndim - 1))
                    elif self._name[p] in ['U', 'V'] and isinstance(gating_type, int):
                        update = update * lr_z[:, None].expand(-1, gating_type).flatten()
                    else:
                        update = lr * update
                p.add_(update)
        return loss


class AdamWithProj(torch.optim.Optimizer):
    def __init__(self, named_params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
        named_params = list(named_params)
        params = [p for _, p in named_params]
        self._name = {p: n for n, p in named_params}
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None, use_proj=False, proj_mtrx_dict=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr, eps = group["lr"], group["eps"]
            beta1, beta2 = group["betas"]
            wd = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                # ===== weight decay =====
                if wd != 0:
                    grad = grad.add(p, alpha=wd)
                # ===== state & moments =====
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = torch.zeros((), dtype=torch.int64)
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)
                state["step"] += 1
                t = state["step"].item()
                m, v = state["exp_avg"], state["exp_avg_sq"]
                m.mul_(beta1).add_(grad, alpha=1 - beta1)
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                bias_c1 = 1 - beta1 ** t
                bias_c2 = 1 - beta2 ** t
                m_hat = m / bias_c1
                v_hat = v / bias_c2
                update = -lr * m_hat / (v_hat.sqrt() + eps)
                if use_proj:
                    pname = self._name[p]
                    if pname == 'Wb_in':
                        update = proj_mtrx_dict['proj_2'] @ update @ proj_mtrx_dict['proj_1']
                    elif pname == 'Wb_out':
                        update = proj_mtrx_dict['proj_4'] @ update @ proj_mtrx_dict['proj_3']
                p.add_(update)
        return loss


class SGDWithProj(torch.optim.Optimizer):
    def __init__(self, named_params, lr=1e-3, momentum=0.0, weight_decay=0.0):
        named_params = list(named_params)
        params = [p for _, p in named_params]
        self._name = {p: n for n, p in named_params}
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None, use_proj=False, proj_mtrx_dict=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            wd = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                # ===== weight decay =====
                if wd != 0:
                    grad = grad.add(p, alpha=wd)
                # ===== momentum =====
                state = self.state[p]
                if momentum != 0:
                    if "momentum_buffer" not in state:
                        buf = state["momentum_buffer"] = torch.clone(grad).detach()
                    else:
                        buf = state["momentum_buffer"]
                        buf.mul_(momentum).add_(grad)
                    update = -lr * buf
                else:
                    update = -lr * grad
                # ===== projection (if enabled) =====
                if use_proj:
                    pname = self._name[p]
                    if pname == 'Wb_in':
                        update = proj_mtrx_dict['proj_2'] @ update @ proj_mtrx_dict['proj_1']
                    elif pname == 'Wb_out':
                        update = proj_mtrx_dict['proj_4'] @ update @ proj_mtrx_dict['proj_3']
                p.add_(update)
        return loss


def cov_to_proj(cov, alpha):
    D, V = torch.linalg.eig(cov)
    print(f'{D.real.min():.4f}, {D.real.max():.4f}', flush=True)
    print(f'{D.imag.min():.4f}, {D.imag.max():.4f}', flush=True)
    D_scaled = alpha / (alpha + D)
    P = V @ torch.diag(D_scaled) @ V.T
    return P.real


@torch.no_grad()
def compute_proj_matrices(model, vl_data_loader, proj_mtrx_dict, itask, alpha_cov):
    device = next(model.parameters()).device
    dim_hid = model.dim_hid
    dim_s, dim_y = vl_data_loader.dataset.dim_s, vl_data_loader.dataset.dim_y
    cur_batch = next(iter(vl_data_loader))
    syz, lengths = cur_batch
    syz = syz.to(device)
    mask = torch.arange(syz.shape[0]).unsqueeze(1) < lengths.unsqueeze(0)  # (seq_len, batch)
    mask = mask.to(device)
    s, y, z = syz[..., :dim_s], syz[..., dim_s:(dim_s + dim_y)], syz[..., (dim_s + dim_y):]
    _, states = model(s, z)
    ones = torch.ones(s.shape[0], s.shape[1], 1, device=s.device)
    vec_1 = torch.cat([s, z, model.nonlin(states), ones], dim=-1)
    compute_covariance = lambda x: x.T @ x / (x.shape[0] - 1)  # or use biased estimate?
    cur_cov_1 = compute_covariance(vec_1[mask])
    cur_cov_2 = compute_covariance(states[mask])
    cur_cov_4 = compute_covariance(y[mask])
    cov_1 = itask / (itask + 1) * proj_mtrx_dict['cov_1'] + cur_cov_1 / (itask + 1)
    cov_2 = itask / (itask + 1) * proj_mtrx_dict['cov_2'] + cur_cov_2 / (itask + 1)
    cov_3 = cov_1[-(dim_hid + 1):, -(dim_hid + 1):]
    cov_4 = itask / (itask + 1) * proj_mtrx_dict['cov_4'] + cur_cov_4 / (itask + 1)
    proj_1 = cov_to_proj(cov_1, alpha_cov)
    proj_2 = cov_to_proj(cov_2, alpha_cov)
    proj_3 = cov_to_proj(cov_3, alpha_cov)
    proj_4 = cov_to_proj(cov_4, alpha_cov)
    proj_mtrx_dict = dict(cov_1=cov_1, cov_2=cov_2, cov_3=cov_3, cov_4=cov_4,
                          proj_1=proj_1, proj_2=proj_2, proj_3=proj_3, proj_4=proj_4)
    return proj_mtrx_dict


def compute_fisher_info(model, vl_data_loader, loss_kwargs):
    device = next(model.parameters()).device
    dim_s, dim_y = vl_data_loader.dataset.dim_s, vl_data_loader.dataset.dim_y
    fisher_info = {n: torch.zeros_like(p) for n, p in model.named_parameters() if p.requires_grad}
    model.eval()
    count = 0
    for ib, cur_batch in enumerate(vl_data_loader):
        syz, lengths = cur_batch
        syz = syz.to(device)
        s, y, z = syz[..., :dim_s], syz[..., dim_s:(dim_s + dim_y)], syz[..., (dim_s + dim_y):]
        out, states = model(s, z)
        loss = masked_mse_loss(out, model.nonlin(states), y, lengths, **loss_kwargs)
        model.zero_grad()
        loss.backward()
        for n, p in model.named_parameters():
            if p.grad is not None:
                fisher_info[n] += p.grad.detach() ** 2
        count += 1
    for n in fisher_info:
        fisher_info[n] /= count
    return fisher_info


def train_leakyrnn_sequential(seed=0, dim_hid=50, dim_s=5, dim_y=3,
                              alpha=0.5, nonlin='tanh', sig_r=0, w_fix=1, n_skip=0, reg_act=0, strict=False,
                              optim='AdamWithProj', reset_optim=True, lr=0.01, weight_decay=0,
                              use_proj=False, use_ewc=False, ewc_lambda=0,
                              batch_size=256, num_iter=500, n_trials_ts=200, n_trials_vl=200,
                              sig_s=0.05, p_stay=0.9, min_Te=5, nx=2, d_stim=np.pi/2,
                              epoch_type=1, fixation_type=1, info_type='c', min_Te_R=None, p_stay_R=None,
                              task_list=['PRO_D', 'PRO_M', 'ANTI_D', 'ANTI_M'], mixed_train=False,
                              verbose=True, ckpt_step=10, alpha_cov=0.001,
                              save_dir=None, retrain=True, save_ckpt=False, save_after_itask=-1, **kwargs):
    set_deterministic(seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = LeakyRNN(dim_i=(dim_s + len(task_list)), dim_y=dim_y, dim_hid=dim_hid,
                     alpha=alpha, nonlin=nonlin, sig_r=sig_r).to(device)

    if save_dir is not None and os.path.isfile(save_dir + '/model.pth') and not retrain:
        save_dict = torch.load(save_dir + '/model.pth', map_location=torch.device(device))
        model.load_state_dict(save_dict)
        tr_loss_arr = np.load(save_dir + '/tr_loss.npy')
        ts_loss_arr = np.load(save_dir + '/ts_loss.npy')
        perf_sfx = '_strict' if strict else ''
        ts_perf_arr = np.load(save_dir + f'/ts_perf{perf_sfx}.npy')
        return model, tr_loss_arr, ts_loss_arr, ts_perf_arr

    print(sum(p.numel() for p in model.parameters() if p.requires_grad), flush=True)
    data_kwargs = dict(dim_s=dim_s, dim_y=dim_y, z_list=None, task_list=task_list,
                       sig_s=sig_s, p_stay=p_stay, min_Te=min_Te, nx=nx, d_stim=d_stim,
                       epoch_type=epoch_type, fixation_type=fixation_type,
                       info_type=info_type, min_Te_R=min_Te_R, p_stay_R=p_stay_R)
    ts_data_loaders = []
    for itask, task in enumerate(task_list):
        ts_dataset = TaskDataset(n_trials=n_trials_ts, task=task, **data_kwargs)
        ts_data_loader = DataLoader(dataset=ts_dataset, batch_size=batch_size, collate_fn=collate_fn)
        ts_data_loaders.append(ts_data_loader)

    loss_kwargs = dict(w_fix=w_fix, n_skip=n_skip, reg_act=reg_act)
    ts_loss, ts_perf, _ = evaluate(model, ts_data_loaders, loss_kwargs, strict=strict)
    tr_loss_arr = [ts_loss[0]]
    ts_loss_arr = [[ts_loss[itask]] for itask in range(len(task_list))]
    ts_perf_arr = [[ts_perf[itask]] for itask in range(len(task_list))]
    ckpt_list = [deepcopy(model.state_dict())]
    ###########################################
    proj_mtrx_dict = dict(cov_1=0, cov_2=0, cov_3=0, cov_4=0, proj_1=None, proj_2=None, proj_3=None, proj_4=None)
    ###########################################
    fisher_total, theta_merged = {}, {}
    ###########################################
    optimizer = eval(optim)(model.named_parameters(), lr=lr, weight_decay=weight_decay)
    for itask, task in enumerate(task_list):
        if mixed_train:
            task = 'all'
        if reset_optim:
            optimizer = eval(optim)(model.named_parameters(), lr=lr, weight_decay=weight_decay)
        for i_iter in range(num_iter):
            tr_dataset = TaskDataset(n_trials=batch_size, task=task, **data_kwargs)
            tr_data_loader = DataLoader(dataset=tr_dataset, batch_size=batch_size, collate_fn=collate_fn)
            model.train()
            for ib, cur_batch in enumerate(tr_data_loader):
                syz, lengths = cur_batch
                syz = syz.to(device)
                s, y, z = syz[..., :dim_s], syz[..., dim_s:(dim_s + dim_y)], syz[..., (dim_s + dim_y):]
                out, states = model(s, z)
                loss = masked_mse_loss(out, model.nonlin(states), y, lengths, **loss_kwargs)
                #####################
                if use_ewc and fisher_total:
                    ewc_loss = 0.0
                    for n, p in model.named_parameters():
                        if n in fisher_total:
                            ewc_loss = ewc_loss + (fisher_total[n] * (p - theta_merged[n]).pow(2)).sum()
                    loss = loss + ewc_lambda * ewc_loss
                #####################
                optimizer.zero_grad()
                loss.backward()
                #####################
                optimizer.step(use_proj=(use_proj & (itask > 0)), proj_mtrx_dict=proj_mtrx_dict)
                #####################
            if (i_iter + 1) % ckpt_step == 0:
                tr_loss_arr.append(loss.item())
                if verbose:
                    print(f'task {task}, iter {i_iter + 1}, train loss: {loss.item():.4f}', flush=True)
                ts_loss, ts_perf, _ = evaluate(model, ts_data_loaders, loss_kwargs, strict=strict)
                for itask_ts, task_ts in enumerate(task_list):
                    ts_loss_arr[itask_ts].append(ts_loss[itask_ts])
                    ts_perf_arr[itask_ts].append(ts_perf[itask_ts])
                    if verbose:
                        print(f'test on task {task_ts}: loss {ts_loss[itask_ts]:.4f}, '
                              f'perf {ts_perf[itask_ts]:.4f}', flush=True)
                if save_ckpt:
                    ckpt_list.append(deepcopy(model.state_dict()))
        ################################################
        if use_proj:
            vl_dataset = TaskDataset(n_trials=n_trials_vl, task=task, **data_kwargs)
            vl_data_loader = DataLoader(dataset=vl_dataset, batch_size=n_trials_vl, collate_fn=collate_fn)
            proj_mtrx_dict = compute_proj_matrices(model, vl_data_loader, proj_mtrx_dict, itask, alpha_cov)
        ################################################
        if use_ewc:
            vl_dataset = TaskDataset(n_trials=n_trials_vl, task=task, **data_kwargs)
            vl_data_loader = DataLoader(dataset=vl_dataset, batch_size=1, collate_fn=collate_fn)
            fisher_new = compute_fisher_info(model, vl_data_loader, loss_kwargs)
            params_new = {n: p.clone().detach() for n, p in model.named_parameters() if p.requires_grad}
            for n in fisher_new:
                if n in fisher_total:
                    f_old = fisher_total[n]
                    theta_old = theta_merged[n]
                    f_new = fisher_new[n]
                    theta_new = params_new[n]
                    fisher_total[n] = f_old + f_new
                    theta_merged[n] = (f_old * theta_old + f_new * theta_new) / (f_old + f_new + 1e-10)
                else:
                    fisher_total[n] = fisher_new[n]
                    theta_merged[n] = params_new[n]
        ################################################
        if itask == save_after_itask and save_dir is not None:
            save_sfx = f'after_itask{itask}'
            torch.save(model.state_dict(), save_dir + f'/model_{save_sfx}.pth')
            if use_proj:
                torch.save(proj_mtrx_dict, save_dir + f'/proj_mtrx_dict_{save_sfx}.pt')
            if use_ewc:
                torch.save(fisher_total, save_dir + f'/fisher_total_{save_sfx}.pt')
                torch.save(theta_merged, save_dir + f'/theta_merged_{save_sfx}.pt')
        ################################################
    tr_loss_arr = np.array(tr_loss_arr)
    ts_loss_arr = np.array(ts_loss_arr)
    ts_perf_arr = np.array(ts_perf_arr)
    if save_dir is not None:
        torch.save(model.state_dict(), save_dir + '/model.pth')
        torch.save(ckpt_list, save_dir + '/ckpt_list.pth')
        np.save(save_dir + '/tr_loss.npy', tr_loss_arr)
        np.save(save_dir + '/ts_loss.npy', ts_loss_arr)
        perf_sfx = '_strict' if strict else ''
        np.save(save_dir + f'/ts_perf{perf_sfx}.npy', ts_perf_arr)
    return model, tr_loss_arr, ts_loss_arr, ts_perf_arr


class NMRNN(nn.Module):
    def __init__(self, dim_s=5, dim_y=3, dim_c=5, dim_z=50, dim_h=50, rank=3, sig_r=0,
                 alpha_z=0.01, alpha_h=0.1, nonlin='tanh', **kwargs):
        super(NMRNN, self).__init__()
        self.dim_z = dim_z
        self.rank = rank
        self.dim_h = dim_h
        self.alpha_z = alpha_z
        self.alpha_h = alpha_h
        self.sig_r = sig_r
        self.W_zz = nn.Parameter(torch.randn(dim_z, dim_z) * math.sqrt(1 / dim_z))
        self.Wz_in = nn.Parameter(torch.randn(dim_z, dim_s + dim_c) * math.sqrt(2 / (dim_z + dim_s + dim_c)))
        self.bz_in = nn.Parameter(torch.zeros(dim_z))
        self.Wz_out = nn.Parameter(torch.randn(rank, dim_z) * math.sqrt(2 / (rank + dim_z)))
        self.bz_out = nn.Parameter(torch.zeros(rank))
        init_scale = (1 / dim_h) ** 0.25 / math.sqrt(rank)
        self.U = nn.Parameter(torch.randn(dim_h, rank) * init_scale)
        self.V = nn.Parameter(torch.randn(dim_h, rank) * init_scale)
        self.W_in = nn.Parameter(torch.randn(dim_h, dim_s) * math.sqrt(2 / (dim_h + dim_s)))
        self.b_in = nn.Parameter(torch.zeros(dim_h))
        self.W_out = nn.Parameter(torch.randn(dim_y, dim_h) * math.sqrt(2 / (dim_h + dim_y)))
        self.b_out = nn.Parameter(torch.zeros(dim_y))
        self.nonlin = getattr(F, nonlin) if nonlin != 'linear' else (lambda a: a)

    def step(self, s_t, c_t, state_z, state_h):
        tmp = self.nonlin(state_z) @ self.W_zz.T + torch.cat([s_t, c_t], dim=-1) @ self.Wz_in.T + self.bz_in
        tmp = tmp + math.sqrt(2 / self.alpha_z) * self.sig_r * torch.randn(*tmp.shape).to(tmp.device)
        state_z = (1 - self.alpha_z) * state_z + self.alpha_z * tmp
        signal_z = torch.sigmoid(state_z @ self.Wz_out.T + self.bz_out)
        tmp = torch.einsum('br,hr,kr,bk->bh', signal_z, self.U, self.V, self.nonlin(state_h))
        tmp = tmp + s_t @ self.W_in.T + self.b_in
        tmp = tmp + math.sqrt(2 / self.alpha_h) * self.sig_r * torch.randn(*tmp.shape).to(tmp.device)
        state_h = (1 - self.alpha_h) * state_h + self.alpha_h * tmp
        output = self.nonlin(state_h) @ self.W_out.T + self.b_out
        return output, state_z, state_h

    def forward(self, s, c):
        seq_len, batch_size, _ = s.shape
        outputs, states_z, states_h = [], [], []
        state_z = torch.zeros(batch_size, self.dim_z, device=s.device)
        state_h = torch.zeros(batch_size, self.dim_h, device=s.device)
        for t in range(seq_len):
            output, state_z, state_h = self.step(s[t], c[t], state_z, state_h)
            outputs.append(output)
            states_z.append(state_z)
            states_h.append(state_h)
        outputs = torch.stack(outputs, dim=0)
        states_z = torch.stack(states_z, dim=0)
        states_h = torch.stack(states_h, dim=0)
        states = torch.cat([states_z, states_h], dim=-1)
        return outputs, states


def train_nmrnn_sequential(seed=0, dim_s=5, dim_y=3, dim_z=50, dim_h=100, rank=5, sig_r=0,
                           alpha_z=0.01, alpha_h=0.1, nonlin='tanh',
                           w_fix=1, n_skip=0, reg_act=0, strict=True,
                           optim='Adam', reset_optim=False, lr=0.01, weight_decay=0,
                           batch_size=256, num_iter=500, n_trials_ts=200,
                           sig_s=0.05, p_stay=0.9, min_Te=5, nx=2, d_stim=np.pi/2,
                           epoch_type=1, fixation_type=1, info_type='c', min_Te_R=None, p_stay_R=None,
                           task_list=['PRO_D', 'PRO_M', 'ANTI_D', 'ANTI_M'], mixed_train=False,
                           verbose=True, ckpt_step=10,
                           save_dir=None, retrain=True, save_ckpt=False, save_after_itask=-1, **kwargs):
    set_deterministic(seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = NMRNN(dim_s=dim_s, dim_y=dim_y, dim_c=len(task_list), dim_z=dim_z, dim_h=dim_h,
                  rank=rank, sig_r=sig_r, alpha_z=alpha_z, alpha_h=alpha_h, nonlin=nonlin).to(device)
    if save_dir is not None and os.path.isfile(save_dir + '/model.pth') and not retrain:
        save_dict = torch.load(save_dir + '/model.pth', map_location=torch.device(device))
        model.load_state_dict(save_dict)
        tr_loss_arr = np.load(save_dir + '/tr_loss.npy')
        ts_loss_arr = np.load(save_dir + '/ts_loss.npy')
        perf_sfx = '_strict' if strict else ''
        ts_perf_arr = np.load(save_dir + f'/ts_perf{perf_sfx}.npy')
        return model, tr_loss_arr, ts_loss_arr, ts_perf_arr

    print(sum(p.numel() for p in model.parameters() if p.requires_grad), flush=True)
    data_kwargs = dict(dim_s=dim_s, dim_y=dim_y, z_list=None, task_list=task_list,
                       sig_s=sig_s, p_stay=p_stay, min_Te=min_Te, nx=nx, d_stim=d_stim,
                       epoch_type=epoch_type, fixation_type=fixation_type,
                       info_type=info_type, min_Te_R=min_Te_R, p_stay_R=p_stay_R)
    ts_data_loaders = []
    for itask, task in enumerate(task_list):
        ts_dataset = TaskDataset(n_trials=n_trials_ts, task=task, **data_kwargs)
        ts_data_loader = DataLoader(dataset=ts_dataset, batch_size=batch_size, collate_fn=collate_fn)
        ts_data_loaders.append(ts_data_loader)

    loss_kwargs = dict(w_fix=w_fix, n_skip=n_skip, reg_act=reg_act)
    ts_loss, ts_perf, _ = evaluate(model, ts_data_loaders, loss_kwargs, strict=strict)
    tr_loss_arr = [ts_loss[0]]
    ts_loss_arr = [[ts_loss[itask]] for itask in range(len(task_list))]
    ts_perf_arr = [[ts_perf[itask]] for itask in range(len(task_list))]
    ckpt_list = [deepcopy(model.state_dict())]

    optimizer = getattr(torch.optim, optim)(model.parameters(), lr=lr, weight_decay=weight_decay)
    for itask, task in enumerate(task_list):
        if mixed_train:
            task = 'all'
        if reset_optim:
            optimizer = getattr(torch.optim, optim)(model.named_parameters(), lr=lr, weight_decay=weight_decay)
        for i_iter in range(num_iter):
            tr_dataset = TaskDataset(n_trials=batch_size, task=task, **data_kwargs)
            tr_data_loader = DataLoader(dataset=tr_dataset, batch_size=batch_size, collate_fn=collate_fn)
            model.train()
            for ib, cur_batch in enumerate(tr_data_loader):
                syz, lengths = cur_batch
                syz = syz.to(device)
                s, y, z = syz[..., :dim_s], syz[..., dim_s:(dim_s + dim_y)], syz[..., (dim_s + dim_y):]
                out, states = model(s, z)
                loss = masked_mse_loss(out, model.nonlin(states), y, lengths, **loss_kwargs)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            if (i_iter + 1) % ckpt_step == 0:
                tr_loss_arr.append(loss.item())
                if verbose:
                    print(f'task {task}, iter {i_iter + 1}, train loss: {loss.item():.4f}', flush=True)
                ts_loss, ts_perf, _ = evaluate(model, ts_data_loaders, loss_kwargs, strict=strict)
                for itask_ts, task_ts in enumerate(task_list):
                    ts_loss_arr[itask_ts].append(ts_loss[itask_ts])
                    ts_perf_arr[itask_ts].append(ts_perf[itask_ts])
                    if verbose:
                        print(f'test on task {task_ts}: loss {ts_loss[itask_ts]:.4f}, '
                              f'perf {ts_perf[itask_ts]:.4f}', flush=True)
                if save_ckpt:
                    ckpt_list.append(deepcopy(model.state_dict()))
        if itask == save_after_itask and save_dir is not None:
            save_sfx = f'after_itask{itask}'
            torch.save(model.state_dict(), save_dir + f'/model_{save_sfx}.pth')
        ################################################
    tr_loss_arr = np.array(tr_loss_arr)
    ts_loss_arr = np.array(ts_loss_arr)
    ts_perf_arr = np.array(ts_perf_arr)
    if save_dir is not None:
        torch.save(model.state_dict(), save_dir + '/model.pth')
        torch.save(ckpt_list, save_dir + '/ckpt_list.pth')
        np.save(save_dir + '/tr_loss.npy', tr_loss_arr)
        np.save(save_dir + '/ts_loss.npy', ts_loss_arr)
        perf_sfx = '_strict' if strict else ''
        np.save(save_dir + f'/ts_perf{perf_sfx}.npy', ts_perf_arr)
    return model, tr_loss_arr, ts_loss_arr, ts_perf_arr


if __name__ == '__main__':
    import argparse
    import platform
    from train_config import load_config

    if platform.system() == 'Windows':
        save_name = 'nmrnn_rk3_PdAdPmAmPdmAdm_mixed_sd0'
        config = load_config(save_name)
        config['retrain'] = False
    else:
        parser = argparse.ArgumentParser()
        parser.add_argument("--save_name", help="save_name", type=str, default="")
        args = parser.parse_args()
        save_name = args.save_name
        config = load_config(save_name)
    train_fn = eval(config.get('train_fn', 'train_cxtrnn_sequential'))
    model, tr_loss_arr, ts_loss_arr, ts_perf_arr = train_fn(**config)
    ####################################################
    fig, axes = plt.subplots(3, 1, figsize=(4, 6), sharex=True, sharey=False)
    axes[0].plot(tr_loss_arr, color='gray')
    axes[0].set_ylabel('training loss')
    for iax in [1, 2]:
        ts_arr = [ts_loss_arr, ts_perf_arr][iax - 1]
        ylabel = ['test loss', 'test perf'][iax - 1]
        for itask in range(len(config['task_list'])):
            axes[iax].plot(ts_arr[itask], color=f'C{itask}', label=config['task_list'][itask])
        axes[iax].legend(loc='center left', bbox_to_anchor=(1, 0.5))
        axes[iax].set_ylabel(ylabel)
        ymin, ymax = axes[iax].get_ylim()
        dn_ckpt = config['num_iter'] / config['ckpt_step']
        for itask in range(len(config['task_list']) + 1):
            axes[iax].plot([itask * dn_ckpt + 1] * 2, [ymin, ymax], alpha=0.2, color='gray')
        for itask in range(len(config['task_list'])):
            axes[iax].fill_between([itask * dn_ckpt + 1, (itask + 1) * dn_ckpt + 1], ymin, ymax, color=f'C{itask}', alpha=0.1)
    if config['save_dir'] is not None:
        fig.savefig(config['save_dir'] + '/loss.png', bbox_inches='tight')
    ####################################################
    dim_s = config.get('dim_s', 3)
    dim_y = config.get('dim_y', 3)
    if config['z_list'] is None:
        config['z_list'] = get_z_list(config['task_list'], config['epoch_type'])
    fig, axes = plt.subplots(dim_s + dim_y, len(config['task_list']),
                             figsize=(2 * len(config['task_list']), dim_s + dim_y),
                             sharey=True, sharex='col')
    if len(axes.shape) == 1:
        axes = axes[:, None]
    for itask, task in enumerate(config['task_list']):
        epoch_str = task_epoch(task, epoch_type=config['epoch_type'])
        x = np.random.randint(config['nx'])
        sig_y = 0
        syz = generate_trial(task, x, config['z_list'], config['task_list'], config['sig_s'], sig_y,
                             config['p_stay'], config['min_Te'], config['d_stim'],
                             config['epoch_type'], config['fixation_type'],
                             info_type=config.get('info_type', 'z'),
                             min_Te_R=config.get('min_Te_R', None), p_stay_R=config.get('p_stay_R', None))
        syz = torch.tensor(syz[:, None, :], dtype=torch.float32).to(next(model.parameters()).device)
        s, y, z = syz[..., :dim_s], syz[..., dim_s:(dim_s + dim_y)], syz[..., (dim_s + dim_y):]
        with torch.no_grad():
            out, _ = model(s, z)
        ########### plot #########
        axes[0, itask].set_title(task)
        for iax in range(dim_s):
            axes[iax, itask].plot(s[:, :, iax].cpu().numpy())
        for iax in range(dim_y):
            axes[iax + dim_s, itask].plot(out[:, :, iax].cpu().numpy())
            axes[iax + dim_s, itask].plot(y[:, :, iax].cpu().numpy())
    if config['save_dir'] is not None:
        fig.savefig(config['save_dir'] + '/example.png', bbox_inches='tight')