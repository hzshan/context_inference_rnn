import os
import torch
import pickle
import torch.nn as nn
import random
from copy import deepcopy
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
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

    def step(self, s_t, z_t, x_state):
        tmp = torch.einsum('br,hr,kr,bk->bh', self.gating(z_t), self.U, self.V, self.nonlin(x_state))
        tmp = tmp + np.sqrt(2 / self.alpha) * self.sig_r * torch.randn(*tmp.shape).to(tmp.device)
        x_state = (1 - self.alpha) * x_state + self.alpha * (tmp + self.input_layer(s_t, z_t))
        output = self.output_layer(self.nonlin(x_state), z_t)
        return output, x_state

    def forward(self, s, z):
        seq_len, batch_size, _ = s.shape
        x_state = torch.zeros(batch_size, self.dim_hid, device=s.device)  # Initialize x state
        outputs = []
        for t in range(seq_len):
            output, x_state = self.step(s[t], z[t], x_state)
            outputs.append(output)
        outputs = torch.stack(outputs, dim=0)
        return outputs


class LeakyRNN(nn.Module):
    def __init__(self, dim_i=3, dim_y=3, dim_hid=50, alpha=0.5, nonlin='tanh',
                 init_scale=0.1, sig_r=0, **kwargs):
        super(LeakyRNN, self).__init__()
        self.W_full = nn.Parameter(torch.randn(dim_hid, dim_i + dim_hid) * init_scale)
        self.b_in = nn.Parameter(torch.randn(dim_hid) * init_scale)
        self.W_out = nn.Parameter(torch.randn(dim_y, dim_hid) * init_scale)
        self.b_out = nn.Parameter(torch.randn(dim_y) * init_scale)
        self.activation = torch.tanh
        self.alpha = alpha
        self.dim_hid = dim_hid
        self.dim_i = dim_i
        self.dim_y = dim_y
        self.nonlin = getattr(F, nonlin) if nonlin != 'linear' else (lambda a: a)
        self.sig_r = sig_r

    def step(self, inp, state):
        tmp = torch.cat([inp, self.nonlin(state)], dim=-1) @ self.W_full.T + self.b_in
        noise = np.sqrt(2 / self.alpha) * self.sig_r * torch.randn(*tmp.shape).to(tmp.device)
        state = (1 - self.alpha) * state + self.alpha * (tmp + noise)
        output = self.nonlin(state) @ self.W_out.T + self.b_out
        return output, state

    def forward(self, s, c):
        inputs = torch.cat([s, c], dim=-1)
        seq_len, batch_size, _ = inputs.shape
        x_state = torch.zeros(batch_size, self.dim_hid, device=inputs.device)  # Initialize x state
        outputs = []
        for t in range(seq_len):
            output, x_state = self.step(inputs[t], x_state)
            outputs.append(output)
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


def generate_trial(task, x, z_list, task_list, sig_s, sig_y, p_stay, min_Te, d_stim,
                   epoch_type, fixation_type, info_type='z', epoch_str=None):
    if epoch_str is None:
        epoch_str = task_epoch(task, epoch_type=epoch_type)
    sy, boundaries = compose_trial(epoch_str, {'theta_task': x}, sig_s, sig_y, p_stay, min_Te, d_stim=d_stim)
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
    def __init__(self, n_trials=64, dim_s=3, dim_y=3, nx=2, sig_s=0.05, sig_y=0,
                 d_stim=np.pi / 2, p_stay=0.9, min_Te=5,
                 task='PRO_D', task_list=['PRO_D', 'PRO_M', 'ANTI_D', 'ANTI_M'],
                 z_list=['F/D', 'S', 'R_P', 'R_M_P', 'R_A', 'R_M_A'],
                 epoch_type=1, fixation_type=1, info_type='z'):
        self.dim_s = dim_s
        self.dim_y = dim_y
        self.trials = []
        self.gdth_z = []
        for i_trial in range(n_trials):
            x = np.random.randint(nx)
            syz = generate_trial(task, x, z_list, task_list, sig_s, sig_y, p_stay, min_Te, d_stim,
                                 epoch_type, fixation_type, info_type=info_type)
            self.trials.append(torch.tensor(syz, dtype=torch.float32))
            self.gdth_z.append(syz[:, (dim_s + dim_y):])

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
def evaluate(model, ts_data_loaders):
    device = next(model.parameters()).device
    ts_loss = []
    for ts_data_loader in ts_data_loaders:
        cur_ts_loss = []
        dim_s = ts_data_loader.dataset.dim_s
        dim_y = ts_data_loader.dataset.dim_y
        for ib, cur_batch in enumerate(ts_data_loader):
            syz, lengths = cur_batch
            syz = syz.to(device)
            s, y, z = syz[..., :dim_s], syz[..., dim_s:(dim_s + dim_y)], syz[..., (dim_s + dim_y):]
            out = model(s, z)
            loss = masked_mse_loss(out, y, lengths)
            cur_ts_loss.append(loss.item())
        ts_loss.append(np.mean(cur_ts_loss))
    return ts_loss


def update_inferred_z(dataset, itask, task_model, use_y=True):
    max_error = []
    dim_s = dataset.dim_s
    dim_y = dataset.dim_y
    for itrial, trial in enumerate(dataset.trials):
        trial_ = (itask, None, {'s': trial[:, :dim_s].numpy(), 'y': trial[:, dim_s:(dim_s + dim_y)].numpy()})
        gamma, _ = task_model.infer(trial_, use_y=use_y, forward_only=True)
        inferred_z = gamma.sum(axis=(0, 1)).T
        trial[:, (dim_s + dim_y):] = torch.tensor(inferred_z)
        ###### check error #######
        gdth_z = dataset.gdth_z[itrial]
        max_error.append(np.max(np.abs(dataset.trials[itrial][:, (dim_s + dim_y):].numpy() - gdth_z)))
    max_error = np.mean(max_error)
    return max_error


def train_cxtrnn_sequential(seed=0, dim_hid=50, dim_s=3, dim_y=3, alpha=0.5,
                         gating_type=3, rank=None, share_io=True, nonlin='tanh', init_scale=0.1, sig_r=0,
                         optim='Adam', reset_optim=True, lr=0.01, weight_decay=0,
                         batch_size=256, num_iter=500, n_trials_ts=200,
                         sig_s=0.05, p_stay=0.9, min_Te=5, nx=2, d_stim=np.pi/2,
                         epoch_type=1, fixation_type=1, info_type='z',
                         task_list=['PRO_D', 'PRO_M', 'ANTI_D', 'ANTI_M'],
                         z_list=['F/D', 'S', 'R_P', 'R_M_P', 'R_A', 'R_M_A'],
                         use_task_model=False, task_model_ntrials=np.inf, frz_io_layer=False,
                         verbose=True, ckpt_step=10,
                         save_dir=None, retrain=True, save_ckpt=False, **kwargs):
    set_deterministic(seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    rank = len(z_list) * gating_type if isinstance(gating_type, int) else rank
    model = CXTRNN(dim_s=dim_s, dim_y=dim_y, dim_z=len(z_list), rank=rank, dim_hid=dim_hid, alpha=alpha,
                   gating_type=gating_type, share_io=share_io, nonlin=nonlin,
                   init_scale=init_scale, sig_r=sig_r).to(device)
    print(sum(p.numel() for p in model.parameters() if p.requires_grad), flush=True)

    if save_dir is not None and os.path.isfile(save_dir + '/model.pth') and not retrain:
        save_dict = torch.load(save_dir + '/model.pth', map_location=torch.device(device))
        model.load_state_dict(save_dict)
        tr_loss_arr = np.load(save_dir + '/tr_loss.npy')
        ts_loss_arr = np.load(save_dir + '/ts_loss.npy')
        return model, tr_loss_arr, ts_loss_arr

    data_kwargs = dict(dim_s=dim_s, dim_y=dim_y, z_list=z_list, task_list=task_list,
                       sig_s=sig_s, p_stay=p_stay, min_Te=min_Te, nx=nx, d_stim=d_stim,
                       epoch_type=epoch_type, fixation_type=fixation_type, info_type=info_type)
    ts_data_loaders = []
    for itask, task in enumerate(task_list):
        ts_dataset = TaskDataset(n_trials=n_trials_ts, task=task, **data_kwargs)
        ts_data_loader = DataLoader(dataset=ts_dataset, batch_size=batch_size, collate_fn=collate_fn)
        ts_data_loaders.append(ts_data_loader)

    ts_loss = evaluate(model, ts_data_loaders)
    tr_loss_arr = [ts_loss[0]]
    ts_loss_arr = [[ts_loss[itask]] for itask in range(len(task_list))]
    ckpt_list = [deepcopy(model.state_dict())]
    ###########################################
    if use_task_model:
        n_epochs_each_task = [len(set(task_epoch(task, epoch_type=epoch_type).split('->'))) for task in task_list]
        w_fixate = make_delay_or_fixation_epoch({}, 1, 0, 0, d_stim=d_stim)
        w_fixate = np.concatenate([w_fixate['s'], w_fixate['y']], axis=1)
        task_model = TaskModel(nc=len(task_list), nz=len(z_list), nx=nx, sigma=sig_s, d=w_fixate.shape[-1],
                               n_epochs_each_task=n_epochs_each_task, w_fixate=w_fixate)
        ts_err_arr = [[] for _ in range(len(task_list))]
        task_model_ckpt_list = []
    ###########################################
    optimizer = getattr(torch.optim, optim)(model.parameters(), lr=lr, weight_decay=weight_decay)
    for itask, task in enumerate(task_list):
        if reset_optim:
            optimizer = getattr(torch.optim, optim)(model.parameters(), lr=lr, weight_decay=weight_decay)
        for iter in range(num_iter):
            tr_dataset = TaskDataset(n_trials=batch_size, task=task, **data_kwargs)
            #################################################
            if use_task_model:
                if batch_size * (iter + 1) <= task_model_ntrials:
                    for trial in tr_dataset.trials:
                        trial_ = (itask, None, {'s': trial[:, :dim_s].numpy(),
                                                'y': trial[:, dim_s:(dim_s + dim_y)].numpy()})
                        task_model.dynamic_initialize_W(trial_)
                        task_model.learn_single_trial(trial_)
                max_error = update_inferred_z(tr_dataset, itask, task_model, use_y=True)
                if verbose and (iter in [0, 1, num_iter - 1]):
                    print(f'task {task}, iter {iter + 1}, max inference error: {max_error:.4f}', flush=True)
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
                out = model(s, z)
                loss = masked_mse_loss(out, y, lengths)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            if (iter + 1) % ckpt_step == 0:
                tr_loss_arr.append(loss.item())
                if verbose:
                    print(f'task {task}, iter {iter + 1}, train loss: {loss.item():.4f}', flush=True)
                # evaluate model on all tasks
                ####################################################
                if use_task_model:
                    for itask_ts, task_ts in enumerate(task_list):
                        ts_dataset = ts_data_loaders[itask_ts].dataset
                        max_error = update_inferred_z(ts_dataset, itask_ts, task_model, use_y=False)
                        ts_err_arr[itask_ts].append(max_error)
                        if verbose:
                            print(f'test max inference error on task {task_ts}: {max_error:.4f}', flush=True)
                ####################################################
                model.eval()
                ts_loss = evaluate(model, ts_data_loaders)
                for itask_ts, task_ts in enumerate(task_list):
                    ts_loss_arr[itask_ts].append(ts_loss[itask_ts])
                    if verbose:
                        print(f'test loss on task {task_ts}: {ts_loss[itask_ts]:.4f}', flush=True)
                if save_ckpt:
                    ckpt_list.append(deepcopy(model.state_dict()))
                    if use_task_model:
                        task_model_ckpt_list.append(deepcopy(task_model))

    tr_loss_arr = np.array(tr_loss_arr)
    ts_loss_arr = np.array(ts_loss_arr)
    if save_dir is not None:
        torch.save(model.state_dict(), save_dir + '/model.pth')
        torch.save(ckpt_list, save_dir + '/ckpt_list.pth')
        np.save(save_dir + '/tr_loss.npy', tr_loss_arr)
        np.save(save_dir + '/ts_loss.npy', ts_loss_arr)
        if use_task_model:
            np.save(save_dir + '/ts_err.npy', ts_err_arr)
            with open(save_dir + '/task_model.pkl', 'wb') as f:
                pickle.dump(task_model, f)
                pickle.dump(task_model_ckpt_list, f)
    return model, tr_loss_arr, ts_loss_arr


class AdamWithProjection(torch.optim.Optimizer):
    def __init__(self, named_params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
        named_params = list(named_params)
        params = [p for _, p in named_params]
        self._name = {p: n for n, p in named_params}
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None, use_proj=False, proj_mtrx_dict=None, clip_norm=None):
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
                # ===== weight‑decay =====
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
                    if self._name[p] == 'W_full':
                        update = proj_mtrx_dict['proj_act'] @ update @ proj_mtrx_dict['proj_i']
                    elif self._name[p] == 'W_out':
                        update = proj_mtrx_dict['proj_o'] @ update @ proj_mtrx_dict['proj_r']
                if clip_norm is not None:
                    if update.norm() > clip_norm:
                        update = update * (clip_norm / update.norm())
                p.add_(update)
        return loss


def cov_to_proj(cov, alpha):
    D, V = torch.linalg.eig(cov)
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
    s, y, z = syz[..., :dim_s], syz[..., dim_s:(dim_s + dim_y)], syz[..., (dim_s + dim_y):]
    sz = torch.cat([s, z], dim=-1)
    seq_len, batch_size, _ = sz.shape
    states = []
    x_state = torch.zeros(batch_size, dim_hid, device=sz.device)  # Initialize x state
    for t in range(seq_len):
        _, x_state = model.step(sz[t], x_state)
        states.append(x_state)
    states = model.nonlin(torch.stack(states, dim=0))
    full_state = torch.cat([sz, states], dim=-1)
    mask = torch.arange(seq_len).unsqueeze(1) < lengths.unsqueeze(0)  # (seq_len, batch)
    mask = mask.to(device)
    compute_covariance = lambda x: x.T @ x / (x.shape[0] - 1)  # or use biased estimate?
    cur_cov_i = compute_covariance(full_state[mask])
    cur_cov_o = compute_covariance(y[mask])
    cur_cov_act = model.W_full @ cur_cov_i @ model.W_full.T
    cov_i = itask / (itask + 1) * proj_mtrx_dict['cov_i'] + cur_cov_i / (itask + 1)
    cov_act = itask / (itask + 1) * proj_mtrx_dict['cov_act'] + cur_cov_act / (itask + 1)
    cov_o = itask / (itask + 1) * proj_mtrx_dict['cov_o'] + cur_cov_o / (itask + 1)
    cov_r = cov_i[-dim_hid:, -dim_hid:]
    proj_act = cov_to_proj(cov_act, alpha_cov)
    proj_i = cov_to_proj(cov_i, alpha_cov)
    proj_o = cov_to_proj(cov_o, alpha_cov)
    proj_r = cov_to_proj(cov_r, alpha_cov)
    proj_mtrx_dict = dict(cov_i=cov_i, cov_act=cov_act, cov_o=cov_o,
                          proj_act=proj_act, proj_i=proj_i, proj_o=proj_o, proj_r=proj_r)
    return proj_mtrx_dict


def train_leakyrnn_sequential(seed=0, dim_hid=50, dim_s=3, dim_y=3,
                             alpha=0.5, nonlin='tanh', init_scale=0.1, sig_r=0,
                             reset_optim=True, use_proj=False, lr=0.01, weight_decay=0, clip_norm=None,
                             batch_size=256, num_iter=500, n_trials_ts=200, n_trials_vl=200,
                             sig_s=0.05, p_stay=0.9, min_Te=5, nx=2, d_stim=np.pi/2,
                             epoch_type=1, fixation_type=1, info_type='c',
                             task_list=['PRO_D', 'PRO_M', 'ANTI_D', 'ANTI_M'],
                             verbose=True, ckpt_step=10, alpha_cov=0.001,
                             save_dir=None, retrain=True, save_ckpt=False, **kwargs):
    set_deterministic(seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = LeakyRNN(dim_i=(dim_s + len(task_list)), dim_y=dim_y, dim_hid=dim_hid, alpha=alpha, nonlin=nonlin,
                     init_scale=init_scale, sig_r=sig_r).to(device)
    print(sum(p.numel() for p in model.parameters() if p.requires_grad), flush=True)

    if save_dir is not None and os.path.isfile(save_dir + '/model.pth') and not retrain:
        save_dict = torch.load(save_dir + '/model.pth', map_location=torch.device(device))
        model.load_state_dict(save_dict)
        tr_loss_arr = np.load(save_dir + '/tr_loss.npy')
        ts_loss_arr = np.load(save_dir + '/ts_loss.npy')
        return model, tr_loss_arr, ts_loss_arr

    data_kwargs = dict(dim_s=dim_s, dim_y=dim_y, z_list=None, task_list=task_list,
                       sig_s=sig_s, p_stay=p_stay, min_Te=min_Te, nx=nx, d_stim=d_stim,
                       epoch_type=epoch_type, fixation_type=fixation_type, info_type=info_type)
    ts_data_loaders = []
    for itask, task in enumerate(task_list):
        ts_dataset = TaskDataset(n_trials=n_trials_ts, task=task, **data_kwargs)
        ts_data_loader = DataLoader(dataset=ts_dataset, batch_size=batch_size, collate_fn=collate_fn)
        ts_data_loaders.append(ts_data_loader)

    ts_loss = evaluate(model, ts_data_loaders)
    tr_loss_arr = [ts_loss[0]]
    ts_loss_arr = [[ts_loss[itask]] for itask in range(len(task_list))]
    ckpt_list = [deepcopy(model.state_dict())]
    ###########################################
    proj_mtrx_dict = dict(cov_i=0, cov_act=0, cov_o=0, proj_act=None, proj_i=None, proj_o=None, proj_r=None)
    ###########################################
    optimizer = AdamWithProjection(model.named_parameters(), lr=lr, weight_decay=weight_decay)
    for itask, task in enumerate(task_list):
        if reset_optim:
            optimizer = AdamWithProjection(model.named_parameters(), lr=lr, weight_decay=weight_decay)
        for i_iter in range(num_iter):
            tr_dataset = TaskDataset(n_trials=batch_size, task=task, **data_kwargs)
            tr_data_loader = DataLoader(dataset=tr_dataset, batch_size=batch_size, collate_fn=collate_fn)
            model.train()
            for ib, cur_batch in enumerate(tr_data_loader):
                syz, lengths = cur_batch
                syz = syz.to(device)
                s, y, z = syz[..., :dim_s], syz[..., dim_s:(dim_s + dim_y)], syz[..., (dim_s + dim_y):]
                out = model(s, z)
                loss = masked_mse_loss(out, y, lengths)
                optimizer.zero_grad()
                loss.backward()
                #####################
                optimizer.step(use_proj=(use_proj & (itask > 0)), proj_mtrx_dict=proj_mtrx_dict, clip_norm=clip_norm)
                #####################
            if (i_iter + 1) % ckpt_step == 0:
                tr_loss_arr.append(loss.item())
                if verbose:
                    print(f'task {task}, iter {i_iter + 1}, train loss: {loss.item():.4f}', flush=True)
                model.eval()
                ts_loss = evaluate(model, ts_data_loaders)
                for itask_ts, task_ts in enumerate(task_list):
                    ts_loss_arr[itask_ts].append(ts_loss[itask_ts])
                    if verbose:
                        print(f'test loss on task {task_ts}: {ts_loss[itask_ts]:.4f}', flush=True)
                if save_ckpt:
                    ckpt_list.append(deepcopy(model.state_dict()))
        ################################################
        if use_proj:
            vl_dataset = TaskDataset(n_trials=n_trials_vl, task=task, **data_kwargs)
            vl_data_loader = DataLoader(dataset=vl_dataset, batch_size=n_trials_vl, collate_fn=collate_fn)
            proj_mtrx_dict = compute_proj_matrices(model, vl_data_loader, proj_mtrx_dict, itask, alpha_cov)
        ################################################################
    tr_loss_arr = np.array(tr_loss_arr)
    ts_loss_arr = np.array(ts_loss_arr)
    if save_dir is not None:
        torch.save(model.state_dict(), save_dir + '/model.pth')
        torch.save(ckpt_list, save_dir + '/ckpt_list.pth')
        np.save(save_dir + '/tr_loss.npy', tr_loss_arr)
        np.save(save_dir + '/ts_loss.npy', ts_loss_arr)
    return model, tr_loss_arr, ts_loss_arr


if __name__ == '__main__':
    import argparse
    import platform
    from train_config import load_config

    if platform.system() == 'Windows':
        save_name = 'leakyrnn_PPAA_DMDM_sigs_minT5_nx2dx2_nitr50_sameopt_sd0'
        config = load_config(save_name)
        config['retrain'] = True
    else:
        parser = argparse.ArgumentParser()
        parser.add_argument("--save_name", help="save_name", type=str, default="")
        args = parser.parse_args()
        save_name = args.save_name
        config = load_config(save_name)
    train_fn = eval(config.get('train_fn', 'train_cxtrnn_sequential'))
    model, tr_loss_arr, ts_loss_arr = train_fn(**config)
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
    dim_s = config.get('dim_s', 3)
    dim_y = config.get('dim_y', 3)
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
                             config['epoch_type'], config['fixation_type'], info_type=config.get('info_type', 'z'))
        syz = torch.tensor(syz[:, None, :], dtype=torch.float32).to(next(model.parameters()).device)
        s, y, z = syz[..., :dim_s], syz[..., dim_s:(dim_s + dim_y)], syz[..., (dim_s + dim_y):]
        with torch.no_grad():
            out = model(s, z)
        ########### plot #########
        axes[0, itask].set_title(task)
        for iax in range(dim_s):
            axes[iax, itask].plot(s[:, :, iax].cpu().numpy())
        for iax in range(dim_y):
            axes[iax + dim_s, itask].plot(out[:, :, iax].cpu().numpy())
            axes[iax + dim_s, itask].plot(y[:, :, iax].cpu().numpy())
    if config['save_dir'] is not None:
        fig.savefig(config['save_dir'] + '/example.png', bbox_inches='tight')