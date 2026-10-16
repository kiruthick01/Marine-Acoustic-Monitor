"""
Neural model + training utilities for Stage 3 (multi-label acoustic scene classification).

Net      log-mel CNN trunk + (optional) 1-D narrowband-LOFAR branch + environment/sensor-health vector, fused into a
         7-way multi-label head and a 5-way vessel-type head.
Norm     input normalisation fitted on training data only.
Data     tensor bundle (mel, vec, lofar, labels, vessel-class) with chronological slicing.
train_net / predict / adabn / calibration helpers.
"""

import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score


class Net(nn.Module):
    def __init__(self, n_vec, n_lab, n_vc, width=24, use_lofar=False):
        super().__init__()
        w = width
        chans = [1, w, 2 * w, 4 * w, 4 * w]
        self.blocks = nn.ModuleList()
        for i in range(4):
            self.blocks.append(nn.Sequential(
                nn.Conv2d(chans[i], chans[i + 1], 3, padding=1, bias=False), nn.BatchNorm2d(chans[i + 1]), nn.ReLU(),
                nn.Conv2d(chans[i + 1], chans[i + 1], 3, padding=1, bias=False), nn.BatchNorm2d(chans[i + 1]), nn.ReLU(),
                nn.MaxPool2d(2)))
        self.vec = nn.Sequential(nn.Linear(n_vec, 32), nn.ReLU())
        d = 4 * w * 2 + 32
        self.use_lofar = use_lofar
        if use_lofar:
            self.lofar = nn.Sequential(
                nn.Conv1d(1, 16, 7, padding=3, bias=False), nn.BatchNorm1d(16), nn.ReLU(), nn.MaxPool1d(3),
                nn.Conv1d(16, 32, 5, padding=2, bias=False), nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(3),
                nn.Conv1d(32, 48, 5, padding=2, bias=False), nn.BatchNorm1d(48), nn.ReLU())
            d += 96
        self.drop = nn.Dropout(0.3)
        self.head = nn.Linear(d, n_lab)
        self.vc_head = nn.Linear(d, n_vc)

    def forward(self, x, v, l=None):
        for b in self.blocks:
            x = b(x)
        parts = [x.mean((2, 3)), x.amax((2, 3)), self.vec(v)]
        if self.use_lofar:
            y = self.lofar(l)
            parts += [y.mean(2), y.amax(2)]
        z = self.drop(torch.cat(parts, dim=1))
        return self.head(z), self.vc_head(z)


class Norm:
    """Fitted on training windows only. LOFAR is already background-whitened, so a fixed scale is used."""

    def __init__(self, mel_ref, vec_ref):
        m = mel_ref.astype(np.float32)
        self.mu = m.mean(axis=(0, 2), keepdims=True)[None]
        self.sd = m.std(axis=(0, 2), keepdims=True)[None] + 1e-3
        self.vmed = np.median(vec_ref, axis=0)
        q75, q25 = np.percentile(vec_ref, [75, 25], axis=0)
        self.vsc = np.maximum((q75 - q25) / 1.349, vec_ref.std(0) * 0.25 + 1e-6)

    def state(self):
        return {k: getattr(self, k) for k in ("mu", "sd", "vmed", "vsc")}

    @classmethod
    def from_state(cls, st):
        n = cls.__new__(cls)
        for k, v in st.items():
            setattr(n, k, v)
        return n

    def mel(self, m):
        return torch.tensor((m.astype(np.float32)[:, None] - self.mu) / self.sd).half()

    def vec(self, v):
        return torch.tensor(np.clip((v - self.vmed) / self.vsc, -10, 10), dtype=torch.float32)

    @staticmethod
    def lofar(l):
        return torch.tensor(np.clip(l.astype(np.float32) / 6.0, -1.0, 5.0))[:, None].half()


class Data:
    def __init__(self, X, V, L, Y, C):
        self.X, self.V, self.L, self.Y, self.C = X, V, L, Y, C

    def __len__(self):
        return len(self.X)

    def sub(self, idx):
        return Data(self.X[idx], self.V[idx], None if self.L is None else self.L[idx], self.Y[idx], self.C[idx])

    def cat(self, other):
        L = None if self.L is None else torch.cat([self.L, other.L])
        return Data(torch.cat([self.X, other.X]), torch.cat([self.V, other.V]), L,
                    torch.cat([self.Y, other.Y]), torch.cat([self.C, other.C]))


def make_data(norm, df, mel, lofar, vec_cols, labels, vclass_idx, use_lofar):
    return Data(norm.mel(mel), norm.vec(df[vec_cols].values), Norm.lofar(lofar) if use_lofar else None,
                torch.tensor(df[labels].values.astype(np.float32)), torch.tensor(vclass_idx(df)))


def augment(x, v, l, gen):
    b = x.shape[0]
    x = x + (torch.rand(b, 1, 1, 1, generator=gen) - 0.5) * 0.6          # +-~3 dB gain / sensitivity drift
    for i in range(b):
        f0 = int(torch.randint(0, x.shape[2] - 8, (1,), generator=gen))
        x[i, :, f0:f0 + int(torch.randint(0, 8, (1,), generator=gen))] = 0.0
        t0 = int(torch.randint(0, x.shape[3] - 12, (1,), generator=gen))
        x[i, :, :, t0:t0 + int(torch.randint(0, 12, (1,), generator=gen))] = 0.0
    if l is not None:
        sh = int(torch.randint(-2, 3, (1,), generator=gen))
        l = torch.roll(l, sh, dims=2) + 0.04 * torch.randn_like(l)
    return x, v + 0.05 * torch.randn_like(v), l


def _fwd(net, d, s, e):
    return net(d.X[s:e].float(), d.V[s:e], None if d.L is None else d.L[s:e].float())


def predict(net, d, bs=256, logits=False):
    net.eval()
    ps, vcs = [], []
    with torch.no_grad():
        for s in range(0, len(d), bs):
            lo, vc = _fwd(net, d, s, s + bs)
            ps.append(lo if logits else torch.sigmoid(lo))
            vcs.append(F.softmax(vc, 1))
    return torch.cat(ps).numpy(), torch.cat(vcs).numpy()


def mean_ap(Y, P):
    return float(np.nanmean([average_precision_score(Y[:, j], P[:, j]) if 0 < Y[:, j].sum() < len(Y) else np.nan
                             for j in range(Y.shape[1])]))


def train_net(net, train, val, epochs, lr, seed, trainable=None, patience=6, bs=96, log=None, wd=1e-3,
              neg_frac=1.0, vc_weight=0.3):
    """AdamW + one-cycle. `neg_frac` < 1 re-draws a fraction of the label-free windows every epoch (calm background is
    abundant), keeping every positive window; early-stops on validation mAP."""
    gen = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    params = [p for n, p in net.named_parameters() if trainable is None or trainable(n)]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=wd)
    any_pos = (train.Y.sum(1) > 0) | (train.C >= 0)
    n_epoch = int(any_pos.sum() + neg_frac * (~any_pos).sum()) if neg_frac < 1 else len(train)
    steps = epochs * int(np.ceil(n_epoch / bs))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps + 5, pct_start=0.2)
    pos = train.Y.sum(0).clamp(min=1)
    pw = torch.clamp(((len(train.Y) - pos) / pos).sqrt(), 1.0, 12.0)
    cc = torch.bincount(train.C[train.C >= 0], minlength=net.vc_head.out_features).float().clamp(min=1)
    cw = (cc.sum() / cc / len(cc)).clamp(0.5, 3.0)
    best, best_state, bad, hist = -1, None, 0, []
    pos_idx = torch.nonzero(any_pos).squeeze(1)
    neg_idx = torch.nonzero(~any_pos).squeeze(1)
    for ep in range(epochs):
        net.train()
        if neg_frac < 1:
            pick = neg_idx[torch.randperm(len(neg_idx), generator=gen)[:int(neg_frac * len(neg_idx))]]
            ep_idx = torch.cat([pos_idx, pick])
            perm = ep_idx[torch.randperm(len(ep_idx), generator=gen)]
        else:
            perm = torch.randperm(len(train), generator=gen)
        tot = 0.0
        for s in range(0, len(perm), bs):
            idx = perm[s:s + bs]
            if len(idx) < 4:
                continue
            xb, vb, lb = augment(train.X[idx].float(), train.V[idx], None if train.L is None else train.L[idx].float(), gen)
            lo, vc = net(xb, vb, lb)
            loss = F.binary_cross_entropy_with_logits(lo, train.Y[idx], pos_weight=pw)
            m = train.C[idx] >= 0
            if m.any():
                loss = loss + vc_weight * F.cross_entropy(vc[m], train.C[idx][m], weight=cw, label_smoothing=0.05)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tot += float(loss) * len(idx)
        score = mean_ap(val.Y.numpy(), predict(net, val)[0])
        hist.append((tot / len(perm), score))
        if log:
            print(f"    ep{ep + 1:02d} loss {tot / len(perm):.3f} val mAP {score:.3f}", flush=True)
        if score > best:
            best, best_state, bad = score, copy.deepcopy(net.state_dict()), 0
        else:
            bad += 1
            if bad >= patience:
                break
    net.load_state_dict(best_state)
    return net, hist, best


def adabn(net, d, bs=256):
    """Unsupervised domain adaptation: re-estimate BatchNorm statistics on unlabeled target windows (no labels, no
    gradient steps). Returns an adapted copy."""
    net = copy.deepcopy(net)
    bns = [m for m in net.modules() if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d))]
    for m in bns:
        m.reset_running_stats()
        m.momentum = None
    net.train()
    with torch.no_grad():
        for s in range(0, len(d), bs):
            _fwd(net, d, s, s + bs)
    net.eval()
    return net


# ------------------------------------------------------------------ calibration
def fit_temperature(logits, y, iters=200):
    """One temperature per label (Platt-style scalar scaling) fitted on validation logits."""
    T = np.ones(logits.shape[1])
    for j in range(logits.shape[1]):
        if y[:, j].sum() < 3:
            continue
        t = torch.nn.Parameter(torch.zeros(1))
        opt = torch.optim.LBFGS([t], lr=0.1, max_iter=iters)
        lg, yy = torch.tensor(logits[:, j], dtype=torch.float32), torch.tensor(y[:, j], dtype=torch.float32)

        def closure():
            opt.zero_grad()
            loss = F.binary_cross_entropy_with_logits(lg / torch.exp(t), yy)
            loss.backward()
            return loss
        opt.step(closure)
        T[j] = float(torch.exp(t))
    return T


def ece(p, y, bins=10):
    """Expected calibration error for one binary label."""
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    e = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            e += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(e)
